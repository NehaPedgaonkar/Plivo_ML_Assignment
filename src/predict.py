# src/predict.py
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os
from postprocess import filter_spans

def bio_to_spans(offsets, label_ids):
    """
    Convert token-level BIO labels and tokenizer offsets into char-level spans.
    offsets: list of (start,end) pairs (one per token)
    label_ids: list of label ids (one per token) aligned with offsets
    returns: list of (start, end, label) char offsets
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # offsets from tokenizer are ints; special tokens often give (0,0)
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        if "-" not in label:
            # unexpected label format, skip
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                # extend current end to this token's end
                current_end = end
            else:
                # I- tag without matching B-: start a new span (robust)
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            # compute spans from token BIO tags + offsets
            spans = bio_to_spans(offsets, pred_ids)

            # strict postprocess: filter invalid PII candidates
            filtered = filter_spans(spans, text)

            # build final ents list with pii flag per assignment rule
            ents = []
            for ent in filtered:
                lab = ent["label"]
                ents.append(
                    {
                        "start": int(ent["start"]),
                        "end": int(ent["end"]),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )

            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()