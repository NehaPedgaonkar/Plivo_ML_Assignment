# src/model.py
from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL

def create_model(model_name: str):
    # Load config and set labels + dropout
    cfg = AutoConfig.from_pretrained(model_name)

    # Assign label maps (important for DistilBERT)
    cfg.num_labels = len(LABEL2ID)
    cfg.id2label = {int(i): lab for i, lab in ID2LABEL.items()}
    cfg.label2id = {lab: int(i) for lab, i in LABEL2ID.items()}

    # Enable dropout for stable training
    if hasattr(cfg, "classifier_dropout"):
        cfg.classifier_dropout = 0.1
    if hasattr(cfg, "dropout"):
        cfg.dropout = 0.1

    # Now create model ONLY using config
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=cfg
    )

    return model
