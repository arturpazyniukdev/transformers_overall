import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from day01_tokenization import MODEL_NAME
from day02_embeddings import get_embeddings
from day03_attention import OUTPUT_DIR

SAMPLE_SIZE = 2000
SEED = 42


def load_data(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.shuffle(seed=SEED).select(range(n))
    return ds.to_pandas()[["text", "label"]]


def main():
    df = load_data()
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: PreTrainedModel = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    X = get_embeddings(df["text"].tolist(), tokenizer, model)
    y = df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {f1:.3f}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "baseline_results.txt").write_text(f"Macro F1: {f1:.3f}\n")


if __name__ == "__main__":
    main()
