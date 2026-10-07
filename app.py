import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from day06_compare import MODEL_DIR


LABEL_MAP = {0: "Negative", 1: "Positive"}

model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model_ft.eval()


def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    result = f"Prediction: {LABEL_MAP[pred]}\n\nProbabilities:\n"
    for i, prob in enumerate(probs):
        result += f"{LABEL_MAP[i]}: {prob.item() * 100:.2f}%\n"
    return result


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a movie review..."),
    outputs=gr.Textbox(label="Result"),
    title="Sentiment Analysis with DistilBERT",
    description="Enter a review and the fine-tuned model predicts its sentiment.",
)

if __name__ == "__main__":
    demo.launch()
