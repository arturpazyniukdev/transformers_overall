from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from day01_tokenization import MODEL_NAME

OUTPUT_DIR = Path("output")


def visualize_attention(tokens, attentions, tokenizer, layer=0, head=0):
    attn = attentions[layer][0, head]
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"attention_layer{layer}_head{head}.png")
    plt.close()


def main():
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: PreTrainedModel = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()
    text = "This movie was absolutely terrible and I hated it"
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    num_heads = outputs.attentions[0].shape[1]
    for head in range(num_heads):
        visualize_attention(tokens, outputs.attentions, tokenizer, layer=5, head=head)


if __name__ == "__main__":
    main()
