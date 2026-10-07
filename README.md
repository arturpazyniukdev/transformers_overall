# Sentiment Analysis with DistilBERT

A 7-day learning project on transformers: from tokenization to a fine-tuned
sentiment classifier with a web demo. Data: 2,000 IMDB reviews
(`stanfordnlp/imdb`), binary labels (0 = Negative, 1 = Positive),
80/20 stratified split with seed 42.

## Structure

| File | What it does |
|------|--------------|
| `day01_tokenization.py` | Tokenizing text with `distilbert-base-uncased` |
| `day02_embeddings.py` | CLS embeddings and cosine similarity |
| `day03_attention.py` | Attention heatmaps for one sentence |
| `day04_baseline.py` | Baseline: frozen DistilBERT embeddings + LogisticRegression |
| `day05_finetune.py` | Fine-tuning DistilBERT for classification (3 epochs) |
| `day06_compare.py` | Fine-tuned vs baseline: metrics, confusion matrix |
| `day07_errors.py` | Error analysis (false positives / false negatives) |
| `app.py` | Gradio demo |

All generated files go to `output/` (not tracked by git):

- `fine_tuned_model/` — fine-tuned model and tokenizer
- `baseline_model.pkl` — baseline classifier
- `comparison_results.txt` — model comparison
- `confusion_matrix_finetuned.png` — confusion matrix
- `error_analysis.txt` — error analysis

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install transformers torch datasets scikit-learn pandas matplotlib seaborn joblib gradio
```

`output/` is not in the repo, so train the models first:

```bash
python day04_baseline.py   # -> output/baseline_model.pkl
python day05_finetune.py   # -> output/fine_tuned_model/
python day06_compare.py    # -> comparison_results.txt, confusion matrix
python day07_errors.py     # -> error_analysis.txt
```

## Demo

```bash
python app.py
```

Open http://127.0.0.1:7860, enter a review and get the prediction with
class probabilities.

## Results

Test set: 400 reviews.

| Model | F1 (macro) | Accuracy |
|-------|-----------:|---------:|
| Fine-tuned DistilBERT | 0.86 | 0.86 |
| Baseline (embeddings + LogReg) | 0.82 | 0.82 |

Improvement: ~4.9% F1.

Results vary slightly between training runs (no torch seed is set).

## Error analysis

56 errors out of 400: 33 false positives, 23 false negatives.

- Error texts are ~40% longer than average (1750 vs 1252 chars). With
  `max_length=128` the model sees only the start of a review and often
  misses the final verdict.
- Mixed reviews (negative overall, but praising some parts) get predicted
  as positive.
- Ironic positive reviews ("guilty pleasure", "ridiculous") get predicted
  as negative.

## Using the model in code

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("output/fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("output/fine_tuned_model")
model.eval()

inputs = tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()  # 0 = Negative, 1 = Positive
```

## Requirements

- Python 3.12
- transformers, torch, datasets, scikit-learn, pandas
- matplotlib, seaborn, joblib
- gradio (for the demo)
