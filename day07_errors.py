import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from day03_attention import OUTPUT_DIR
from day04_baseline import SEED, load_data
from day05_finetune import MODEL_DIR
from day06_compare import predict_fine_tuned


def build_test_df() -> pd.DataFrame:
    model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model_ft.eval()

    df = load_data()
    texts, labels = df["text"].tolist(), df["label"].tolist()
    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    y_pred_ft = [p["prediction"] for p in predict_fine_tuned(test_texts, model_ft, tokenizer)]

    return pd.DataFrame({"text": test_texts, "true_label": test_labels, "pred_label": y_pred_ft})


def print_examples(title: str, rows: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{title}")
    for _, row in rows.head(n).iterrows():
        print(f"\nText: {row['text'][:100]}...")
        print(f"True: {row['true_label']}, Predicted: {row['pred_label']}")


def format_examples(title: str, rows: pd.DataFrame, n: int = 5) -> str:
    lines = [title]
    for _, row in rows.head(n).iterrows():
        lines.append(f"\nText: {row['text']}")
        lines.append(f"True: {row['true_label']}, Predicted: {row['pred_label']}")
    return "\n".join(lines)


def main():
    df_test = build_test_df()

    errors = df_test[df_test["true_label"] != df_test["pred_label"]].copy()
    fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]
    fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

    print(f"Total errors: {len(errors)}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")

    print_examples("=== FALSE POSITIVES (said good, was bad) ===", fp)
    print_examples("=== FALSE NEGATIVES (said bad, was good) ===", fn)

    errors["text_length"] = errors["text"].str.len()
    print(f"\nAvg length of error texts: {errors['text_length'].mean():.0f}")
    print(f"Avg length of all texts: {df_test['text'].str.len().mean():.0f}")

    observations = (
        "- Error texts are ~40% longer than average; max_length=128 cuts most of the review,\n"
        "  often the final verdict.\n"
        "- Mixed tone: negative reviews with praise for parts (jokes, budget) -> false positive.\n"
        "- Irony / 'guilty pleasure': positive reviews full of negative words\n"
        "  ('ridiculous', 'dumb') -> false negative.\n"
        "- More FP (33) than FN (23): the model leans toward 'positive'."
    )

    report = "\n".join(
        [
            "=== ERROR ANALYSIS ===\n",
            f"Total errors: {len(errors)}",
            f"False Positives: {len(fp)}",
            f"False Negatives: {len(fn)}",
            f"Avg length of error texts: {errors['text_length'].mean():.0f}",
            f"Avg length of all texts: {df_test['text'].str.len().mean():.0f}\n\n",
            format_examples("=== FALSE POSITIVE EXAMPLES ===", fp),
            "\n\n" + format_examples("=== FALSE NEGATIVE EXAMPLES ===", fn),
            "\n\n=== OBSERVATIONS ===",
            observations,
        ]
    )
    (OUTPUT_DIR / "error_analysis.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
