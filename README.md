# Psycholinguistic Manipulation Detector

## Problem
Text manipulation is everywhere - political speeches, fake news, propoganda. This project automatically classifies text by manipulation technique, helping redears identyify when they are being manipulated.

## Classes

| Class | Description | Example |
|-------|-------------|---------|
| `fear_appeal` | Appeals to fear, threats | "Enemies will destroy everything we love!" |
| `emotional_manipulation` | Emotional pressure bypassing logic | "How dare you ignore innocent suffering!" |
| `demagogy_tricks` | False dilemmas, label switching | "Either you're with us or you're a traitor" |
| `authority_appeal` | Appeal to authority without evidence | "Harvard experts confirm this is the only way" |
| `rational_argument` | Facts, logic, sources | "Inflation decreased 2.3% according to Fed data" |

## Architecture
```
Data Collection -> Weak Labeling -> Feature Extraction -> Model Training -> API
```

**Pipeline:**
- 'src/preprocessing/build_dataset.py'
- 'src/features/extractor.py' 
- 'notebooks/04_baseline.ipynb'
- 'notebooks/05_bert.ipynb'
- 'api/main.py'

## Results

| Model | F1-macro | Notes |
|-------|----------|-------|
| TF-IDF + LogReg | 0.67 | Baseline, fast |
| DistilBERT (fine-tuned) | 0.77 | Best model, 6 epochs |

Target metric: F1-macro ≥ 0.70 

## Dataset
- **Sources:** BuzzFeed News, PolitiFact, Liar Dataset (train/valid/test)
- **Size:** 4,663 labeled examples
- **Labeling strategy:** Weak supervision - keyword-based labeling by content, not source

**Why weak labeling?**
Instead of manually labeling 10k texts, we used psycholinguistic keyword lists to assign labels based on what the text *contains*, not where it came from. This give more accurate labels, than mapping entire datasets to single classes.

## Tutorial

### Local

### Docker

## API Usage

## Psycholinguistic Features

## Experiments

All experiments tracked with MLflow:
```bash
mlflow ui
# open http://localhost:5000
```

## Project Structure
```
psycholinguistic-detector/
├── data/
│   ├── raw/          # Original datasets
│   ├── processed/    # Combined labeled dataset
│   └── labeled/      # Manual annotations
├── notebooks/
│   ├── 03_features.ipynb   # Feature extraction experiments
│   ├── 04_baseline.ipynb   # TF-IDF baseline
│   ├── 05_bert.ipynb       # DistilBERT fine-tuning
│   └── 06_mlflow.ipynb     # Experiment tracking
├── src/
│   ├── preprocessing/
│   │   ├── build_dataset.py  # Data pipeline
│   │   └── label_mapper.py   # Label mapping utilities
│   └── features/
│       └── extractor.py      # Psycholinguistic feature extraction
├── api/
│   └── main.py         # FastAPI server
├── models/             # Saved models (not in git)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Stack

- **NLP:** spaCy, HuggingFace, Transformers, DistilBERT
- **ML:** scikit-learn, PyTorch
- **Experiment tracking:** MLflow
- **API:** FastAPI, Uvicorn
- **Containerization:** Docker

## Roadmap
