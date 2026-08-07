import json

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/onnx_quantized")
session = ort.InferenceSession(
    "models/onnx_quantized/model_quantized.onnx", providers=["CPUExecutionProvider"]
)

with open("models/classes.json") as f:
    classes = json.load(f)


def predict(text: str) -> tuple[str, float]:
    enc = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="np"
    )

    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    logits = session.run(None, inputs)[0][0]

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


samples = [
    "Enemies want to destroy everything we love! Our children are in danger!",
    "According to Federal Reserve data, inflation decreased by 2.3 percent",
]

for text in samples:
    label, conf = predict(text)
    print(f"{label:25s} {conf:.3f} <- {text[:50]}")
