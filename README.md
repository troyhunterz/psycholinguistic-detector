# Psycholinguistic Manipulation Detector 

A machine learning tool that detects rhetorical manipulation in text: political speeches, news articles, social media posts, and more.

[![CI](https://github.com/troyhunterz/psycholinguistic-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/troyhunterz/psycholinguistic-detector/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![spaCy](https://img.shields.io/badge/spaCy-3.7.4-blue)

---

## Live demo

**https://psycholinguistic-detector.onrender.com**

Paste any text, then **Quick Classify** (fast) or **Deep Analysis (AI)** for an explanation. Hosted on a free tier, so the first request after it's been idle can take up to a minute while the service wakes up.

## What It Does

The model classifies text into one of five rhetorical categories:

| Label | Description |
| --- | --- |
| 'fear_appeal' | Threats, catastrophizing, enemy imagery |
| 'emotional_manipulation' | Bypasses logic by exploiting emotions |
| 'demagogy_tricks' | False dilemmas, populism, label-switching |
| 'authority_appeal' | Cites authority without real evidence |
| 'rational_argument' | Logic, facts, evidence, cited sources |

There are two analysis modes:
- **Quick Classify** - fast DistilBERT inference, returns label + confidence scores
- **Deep Analysis (AI)** - same classification + a natural language explanation via Groq (llama 3.3 70b)

For deployment, the fine-tuned DistilBERT is exported to ONNX and quantized to int8 (about 4× smaller, ~64 MB) and served with onnxruntime instead of PyTorch. That keeps the container light enough to run on a free 512 MB instance.

---

## Project Structure

```
├── api/
│   ├── main.py                  # FastAPI app: /predict, /analyze, /health, /ready
│   └── static/index.html        # frontend UI
├── config.py                    # settings from environment
├── logging_config.py            # structured JSON logging
├── src/
│   ├── features/extractor.py         # spaCy psycholinguistic features
│   ├── explainer/llm_explainer.py    # Groq LLM for deep analysis
│   └── preprocessing/                # dataset build + label mapping
├── scripts/                     # ONNX export, quantization, verification
├── models/
│   ├── onnx_quantized/          # served int8 ONNX model + tokenizer
│   └── classes.json
├── tests/
├── Dockerfile
├── requirements.txt             # runtime deps (lightweight, no torch)
├── requirements-dev.txt         # + training / export deps
└── .github/workflows/ci.yml
```

## Results

| Model | F1-macro | Notes |
|-------|----------|-------|
| TF-IDF + LogReg | 0.67 | Baseline, fast |
| DistilBERT (fine-tuned) | 0.77 | Best model, 6 epochs |

Target metric: F1-macro ≥ 0.70 

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/troyhunterz/psycholinguistic-detector.git
cd psycholinguistic-detector
python -m venv venv
venv\Scripts\activate
```

Run the service:   `pip install -r requirements.txt`
Train | develop:   `pip install -r requirements-dev.txt`

### 2. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

### 3. The model

The quantized ONNX model (`models/onnx_quantized/`) and `models/classes.json` already ship in the repo, so the service runs out of the box. To rebuild them from scratch, see [Training](#training).

### 4. Run the server

```bash
uvicorn api.main:app --reload
```

Open [http://localhost:8000](https://localhost:8000) in your browser.

---

## API

### `GET /health`

Return server status.
```json
{ "status": "healthy" }
```

### `GET /ready`

Readiness check - confirms the model is loaded. Returns `503` if it is not.
​```json
{ "status": "ready" }
​```

### `POST /predict`
Fast classification. Rate limited to 90 requests/minute

**Request:**
```json
{ "text": "Enemies want to destroy everything we love" }
```

**Response:**
```json
{
  "label": "fear_appeal",
  "confidence": 0.9784,
  "all_scores": {
    "authority_appeal": 0.0016,
    "demagogy_tricks": 0.0024,
    "emotional_manipulation": 0.0157,
    "fear_appeal": 0.9784,
    "rational_argument": 0.0019
  }
}
```

### `POST /analyze`
Classification + LLM explanation. Rate limited to 30 requests/minute.
Requires `GROQ_API_KEY`.

**Response** adds:
```json
 {
    "explanation": "This text belongs to the \"fear_appeal\" class because it employs a threat to evoke fear. The phrase \"'Enemies want to destroy everything we love'\" signals this by using catastrophic language..."
}
```

Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Training

### 1. Build the dataset

Download the BuzzFeed and PolitiFact datasets and place CSVs in `data/raw`, then:

```bash
python src/preprocessing/build_dataset.py
```

This merges news datasets with the LIAR dataset, applies weak labeling heuristics, and outputs a balanced `data/processed/combined_en.csv`.

### 2. Train the model

Fine-tune DistilBERT on the processed dataset (training script not shown: outputs `models/distilbert_best.pt` and `models/classes.json`).

---

## Feature Extraction

`src/features/extractor.py` computes 10 psycholinguistic features per text using spaCy:

| Feature | Description |
| --- | --- |
| `we_ratio` | Proportion of "we/our/us" |
| `they_ratio` | Proportion of "they/enemy" |
| `exclaim_ratio` | Exclamation sentences ratio |
| `question_ratio` | Question sentences ratio |
| `modal_ratio` | Modal verbs ratio |
| `logic_count` | Logical connectors count |
| `adj_ratio` | Adjective ratio |
| `verb_ratio` | Verb ratio |
| `avg_sent_len` | Average sentence length |
| `caps_ratio` | ALL CAPS words ratio |

---

## Tests

```bash
pytest tests/ -v
```

The test suite covers API endpoints (`/health`, `/predict`, `/analyze`) and feature extraction.
The service loads the quantized ONNX model that ships in the repo, so tests run against the real model.

---

## CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. Installs the runtime dependencies
2. Lints with ruff
3. Runs the test suite against the committed ONNX model

---

## Rate Limits

| Endpoint | Limit |
| --- | --- |
| `/predict` | 90 requests / minute |
| `/analyze` | 30 requests / minute |

---

## Roadmap

- [x] Web interface for real-time text analysis
- [x] LLM explanation (GPT/Claude explains why text is manipulative)

## License

MIT
