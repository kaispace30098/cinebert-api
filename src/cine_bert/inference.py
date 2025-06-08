# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model and tokenizer
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# Define inference function
def predict_sentiment(text):
    # Preprocess input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()

    # Return result
    return {
        "input": text,
        "predicted_class": predicted_class,
        "probabilities": probs.squeeze().tolist()
    }

# 🔁 Test inference
if __name__ == "__main__":
    test_texts = [
        "This movie was absolutely fantastic!",
        "The film was boring and too long.",
        "I really enjoyed the plot and the characters.",
        "Terrible acting. Would not recommend.",
        "Went to see the movie with my nephew, I grew up on the movies and shows, and this movie had some good changes and bad changes but I'd have to say more bad than good. I can't lie I still got a few laughs out of the movie, but they removed some of the more iconic moments and some of the changes made no sense. It felt rushed but also dragged on a lot longer than the original, they rushed the iconic scenes and dragged out the new scenes, the new neighbor character is supposed to be the responsibile adult in lilo and nanis life but ends up being the direct reason lilo gets taken by child services. David was barely in the movie and doesn't really feel like a love interest for nani he comes off as an awkward guy the first time we see him."
    ]

    for text in test_texts:
        result = predict_sentiment(text)
        print(result)
