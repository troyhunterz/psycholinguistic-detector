import json
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.explainer.llm_explainer import explain as llm_explain

app = FastAPI(
    title='Psycholinguistic Manipulation Detector',
    description='Detects manipulation techniques in text',
    version='1.0.0'
)

device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('models/tokenizer')

model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=5,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load(
    'models/distilbert_best.pt', map_location=device, weights_only=True))
model.to(device)

model.eval()

with open('models/classes.json') as f:
    classes = json.load(f)

print(f'Model loaded! Classes: {classes}')


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    label: str
    confidence: float
    all_scores: dict


class AnalysisOutput(BaseModel):
    label: str
    confidence: float
    all_scores: dict
    explanation: str


@app.get('/')
def root():
    return {'status': 'ok', 'message': 'Manipulation Detector API'}


@app.get('/health')
def health():
    return {'status': 'healthy'}


@app.post('/predict', response_model=PredictionOutput)
def predict(input: TextInput):
    encoding = tokenizer(
        input.text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )

    probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_label = classes[pred_idx]
    confidence = probs[pred_idx].item()

    all_scores = {
        classes[i]: round(probs[i].item(), 4)
        for i in range(len(classes))
    }

    return PredictionOutput(
        label=pred_label,
        confidence=round(confidence, 4),
        all_scores=all_scores
    )


@app.post('/analyze', response_model=AnalysisOutput)
def analyze(input: TextInput):
    encoding = tokenizer(
        input.text,
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_label = classes[pred_idx]
    confidence = probs[pred_idx].item()
    all_scores = {
        classes[i]: round(probs[i].item(), 4)
        for i in range(len(classes))
    }

    explanation = llm_explain(input.text, pred_label, confidence)

    return AnalysisOutput(
        label=pred_label,
        confidence=round(confidence, 4),
        all_scores=all_scores,
        explanation=explanation
    )
