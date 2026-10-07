import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from day01_tokenization import MODEL_NAME, tokenize_texts


def get_embeddings(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenize_texts(batch_texts, tokenizer)
        with torch.no_grad():
            outputs = model(**tokens)
        cls = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls.cpu().numpy())
    return np.vstack(all_embeddings)


def similarity(
    text1: str, text2: str, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel
) -> float:
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])
    return float(sim[0][0])


def main():
    texts = [
        "This movie was absolutely amazing!",
        "Terrible movie, waste of time.",
        "Pretty good, I liked it.",
        "Boring and too long.",
    ]
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: PreTrainedModel = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embeddings = get_embeddings(texts, tokenizer, model)
    print(embeddings.shape)

    sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
    sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)
    print(f"Similar: {sim1:.3f}")
    print(f"Different: {sim2:.3f}")


if __name__ == "__main__":
    main()
