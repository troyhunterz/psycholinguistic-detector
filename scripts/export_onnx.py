import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5, ignore_mismatched_sizes=True
)

model.load_state_dict(
    torch.load("models/distilbert_best.pt", map_location="cpu", weights_only=True)
)

model.save_pretrained("models/hf_model")

AutoTokenizer.from_pretrained("models/tokenizer").save_pretrained("models/hf_model")

print("Saved HF model to models/hf_model")
