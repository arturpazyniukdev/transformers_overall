import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from day01_tokenization import MODEL_NAME
from day02_embeddings import get_embeddings
from day03_attention import OUTPUT_DIR
from day04_baseline import BASELINE_PATH, SEED, load_data
from day05_finetune import MODEL_DIR

EXAMPLES = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long.",
]


def predict_baseline(
    texts: str | list[str],
    model: LogisticRegression,
    tokenizer: PreTrainedTokenizerBase,
    encoder: PreTrainedModel,
) -> list[dict]:
    if isinstance(texts, str):
        texts = [texts]
    X = get_embeddings(texts, tokenizer, encoder)
    predictions = model.predict(X)
    probs = model.predict_proba(X)
    results = []
    for i, text in enumerate(texts):
        results.append({"text": text, "prediction": int(predictions[i]), "probabilities": probs[i]})
    return results


def predict_fine_tuned(
    texts: str | list[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> list[dict]:
    if isinstance(texts, str):
        texts = [texts]
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        predictions.append({"text": text, "prediction": pred, "probabilities": probs[0].numpy()})
    return predictions


def main():
    model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model_ft.eval()
    baseline_model = joblib.load(BASELINE_PATH)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    encoder.eval()

    preds_ft = predict_fine_tuned(EXAMPLES, model_ft, tokenizer)
    preds_base = predict_baseline(EXAMPLES, baseline_model, tokenizer, encoder)
    for i, text in enumerate(EXAMPLES):
        print(f"\nText: {text}")
        print(f"Fine-tuned: {preds_ft[i]['prediction']} (probs: {preds_ft[i]['probabilities']})")
        print(
            f"Baseline:   {preds_base[i]['prediction']} (probs: {preds_base[i]['probabilities']})"
        )
        print(f"Match: {preds_ft[i]['prediction'] == preds_base[i]['prediction']}")

    df = load_data()
    texts, labels = df["text"].tolist(), df["label"].tolist()
    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    y_pred_ft = [p["prediction"] for p in predict_fine_tuned(test_texts, model_ft, tokenizer)]
    cm_ft = confusion_matrix(test_labels, y_pred_ft)
    print(cm_ft)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ft, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Fine-tuned Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(OUTPUT_DIR / "confusion_matrix_finetuned.png")
    plt.close()

    print("=== Fine-tuned Model ===")
    print(classification_report(test_labels, y_pred_ft))
    f1_ft = f1_score(test_labels, y_pred_ft, average="macro")
    acc_ft = accuracy_score(test_labels, y_pred_ft)

    preds_base_all = predict_baseline(test_texts, baseline_model, tokenizer, encoder)
    y_pred_base = [p["prediction"] for p in preds_base_all]
    print("\n=== Baseline Model ===")
    print(classification_report(test_labels, y_pred_base))
    f1_base = f1_score(test_labels, y_pred_base, average="macro")
    acc_base = accuracy_score(test_labels, y_pred_base)

    improvement = (f1_ft - f1_base) / f1_base * 100
    print("\n=== Comparison ===")
    print(f"Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")
    print(f"Baseline   F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}")
    print(f"F1 improvement: {improvement:.2f}%")

    (OUTPUT_DIR / "comparison_results.txt").write_text(
        "=== Model comparison ===\n\n"
        "Fine-tuned Model:\n"
        f"  F1 (macro): {f1_ft:.4f}\n"
        f"  Accuracy: {acc_ft:.4f}\n"
        "\nBaseline Model:\n"
        f"  F1 (macro): {f1_base:.4f}\n"
        f"  Accuracy: {acc_base:.4f}\n"
        f"\nImprovement: {improvement:.2f}%\n"
    )


if __name__ == "__main__":
    main()
