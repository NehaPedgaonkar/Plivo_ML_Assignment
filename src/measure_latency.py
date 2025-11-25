import argparse
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    # ---------------------------
    # Performance Optimizations
    # ---------------------------
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()  # disable dropout

    device = torch.device("cpu")
    model.to(device)

    # Load data
    data = load_data(args.input)

    # ---------------------------
    # Pre-tokenize all inputs
    # ---------------------------
    encoded_inputs = []
    for ex in data:
        inp = tokenizer(ex["text"], return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        encoded_inputs.append(inp)

    # ---------------------------
    # Warm-up runs (very important)
    # ---------------------------
    for _ in range(5):
        _ = model(**encoded_inputs[0])

    # ---------------------------
    # Measure latency
    # ---------------------------
    latencies = []
    for i in range(args.runs):
        inp = encoded_inputs[i % len(encoded_inputs)]

        start = time.time()
        _ = model(**inp)
        end = time.time()

        latencies.append((end - start) * 1000.0)  # ms

    latencies.sort()
    p50 = latencies[int(0.50 * len(latencies))]
    p95 = latencies[int(0.95 * len(latencies))]

    print(f"Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")

if __name__ == "__main__":
    main()
