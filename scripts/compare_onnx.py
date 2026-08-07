import json

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/onnx_quantized")

with open("models/classes.json") as f:
    classes = json.load(f)

session_fp32 = ort.InferenceSession("models/onnx/model.onnx", providers=["CPUExecutionProvider"])

session_int8 = ort.InferenceSession(
    "models/onnx_quantized/model_quantized.onnx", providers=["CPUExecutionProvider"]
)


def predict(session, text):
    enc = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="np"
    )

    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    logits = session.run(None, inputs)[0][0]
    exp = np.exp(logits)
    probs = exp / exp.sum()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


samples = [
    "Enemies want to destroy everything we love! Our children are in danger!",
    "According to Federal Reserve data, inflation decreased by 2.3 percent.",
    "You must act now before it is too late, there is no other choice.",
]

for text in samples:
    label32, conf32 = predict(session_fp32, text)
    label8, conf8 = predict(session_int8, text)
    match = "OK" if label32 == label8 else "DIFF"
    print(f"[{match}] fp32={label32}({conf32:.3f})  int8={label8}({conf8:.3f})  <-  {text[:45]}")
