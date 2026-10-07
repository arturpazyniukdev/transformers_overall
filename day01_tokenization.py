from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODEL_NAME = "distilbert-base-uncased"


def tokenize_texts(texts, tokenizer: PreTrainedTokenizerBase, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(text, tokens, ids, len(tokens))


def main():
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = ["This movie was great!", "Terrible movie, waste of time."]
    batch = tokenize_texts(texts, tokenizer)
    print(batch["input_ids"].shape)
    print(batch["attention_mask"])
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    explain_tokenization("Transformers are amazing!", tokenizer)


if __name__ == "__main__":
    main()
