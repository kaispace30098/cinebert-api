import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(
    title="Movie Sentiment Classifier",
    version="1.0.0",
    description="Sentiment prediction (positive/negative) for movie reviews using DistilBERT",
)

# Enable CORS if frontend is calling across origins (specify domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model: expects a single "text" field
class SentimentRequest(BaseModel):
    text: str

# MODEL_DIR points to api/saved_model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_DIR = os.path.abspath(MODEL_DIR)

# Automatically use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try loading tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_DIR}: {e}")

def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()[0]
        predicted_class = int(torch.argmax(logits, dim=-1).cpu().item())
    return {"predicted_class": predicted_class, "probabilities": probs}

@app.get("/health")
async def health_check():
    files = os.listdir(MODEL_DIR)
    return {
        "status": "ok",
        "device": str(device),
        "model_dir": MODEL_DIR,
        "files": files,
    }

@app.post("/predict")
async def predict(req: SentimentRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text must not be empty.")
    result = predict_sentiment(text)
    return {
        "input": text,
        "predicted_class": result["predicted_class"],
        "probabilities": {
            "negative": result["probabilities"][0],
            "positive": result["probabilities"][1],
        },
    }
