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
- 'src/preprocessing/build_dataset.py' - data collection and weak labeling
- 'src/features/extractor.py' - 10 psycholinguistic features (we/they ratio, exclamations, modal verbs, etc.)
- 'notebooks/04_baseline.ipynb' - TF-IDF + LogisticRegression baseline
- 'notebooks/05_bert.ipynb' - DistilBERT fine-tuning
- 'api/main.py' FastAPI inference server

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
```bash
# Clone repo
git clone https://github.com/troyhunterz/psycholinguistic-detector
cd psycholinguistic-detector

# Setup environment
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload
```

### Docker
```bash
docker build -t manipulation-detector
docker run -p 8000:8000 manipulation-detector
```

API available at 'http://localhost:8000'
Docs available at 'http://localhost:8000/docs'

## API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Enemies want to destroy everything we love!"}'
```

Response:
```json
{
    "label": "fear_appeal",
    "confidence": 0.9843,
    "all_scores": {
    "authority_appeal": 0.0014,
    "demagogy_tricks": 0.002,
    "emotional_manipulation": 0.0107,
    "fear_appeal": 0.9843,
    "rational_argument": 0.0017
    }
}
```

## Psycholinguistic Features
The feature extractor(`src/features/extractor.py') computes 10 linguistic signals:

| Feature | Description | Manipulation signal |
|---------|-------------|---------------------|
| `we_ratio` | Proportion of "we/our/us" | In-group construction |
| `they_ratio` | Proportion of "they/enemy" | Out-group construction |
| `exclaim_ratio` | Exclamation sentences ratio | Emotional amplification |
| `question_ratio` | Question sentences ratio | Rhetorical questions |
| `modal_ratio` | Modal verbs ratio | False obligation |
| `logic_count` | Logical connectors count | Rational argumentation |
| `adj_ratio` | Adjective ratio | Emotional loading |
| `verb_ratio` | Verb ratio | Action orientation |
| `avg_sent_len` | Average sentence length | Complexity |
| `caps_ratio` | ALL CAPS words ratio | Emphasis/shouting |

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

- [ ] Web interface for real-time text analysis
- [ ] LLM explanation (GPT/Claude explains why text is manipulative)
- [ ] Ukrainian, russian, language support