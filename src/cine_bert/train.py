import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1️⃣ Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2️⃣ Load & split dataset
ds = load_dataset("imdb")
train_ds = ds["train"]
full_test = ds["test"].shuffle(seed=42)
test_ds = full_test.select(range(100))
val_ds = full_test.select(range(100, 2600))

# 3️⃣ Tokenizer & preprocessing function
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(batch):
    tokenized = tokenizer(batch["text"], truncation=True, padding=False)
    tokenized["labels"] = batch["label"]
    return tokenized

# Apply preprocessing and remove original text column
train_ds = train_ds.map(preprocess, batched=True, remove_columns=["text"])
val_ds = val_ds.map(preprocess, batched=True, remove_columns=["text"])
test_ds = test_ds.map(preprocess, batched=True, remove_columns=["text"])

# Set dataset format for PyTorch
for split in (train_ds, val_ds, test_ds):
    split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4️⃣ DataLoader & Collator
collator = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collator)

# 5️⃣ Model, optimizer, scheduler
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 5  # 5 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 6️⃣ Evaluation metric function
def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# For tracking metrics per epoch
history = {
    "epoch": [],
    "train_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": [],
}

# 7️⃣ Training & validation loop
model.train()
for epoch in range(1, 6):  # Epochs 1 to 5
    total_loss = 0.0

    # —— Training ——  
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} train loss: {avg_train_loss:.4f}")

    # —— Validation ——  
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    metrics = compute_metrics(all_labels, all_preds)
    print(f"Epoch {epoch} val metrics: {metrics}")

    # Record history
    history["epoch"].append(epoch)
    history["train_loss"].append(avg_train_loss)
    history["val_accuracy"].append(metrics["accuracy"])
    history["val_precision"].append(metrics["precision"])
    history["val_recall"].append(metrics["recall"])
    history["val_f1"].append(metrics["f1"])

    model.train()

# 8️⃣ Plot and save validation metrics (2×2 plot)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Accuracy", "Precision", "Recall", "F1"),
    shared_xaxes=True
)

# Accuracy
fig.add_trace(
    go.Scatter(x=history["epoch"], y=history["val_accuracy"],
               mode="lines+markers", name="Val Accuracy"),
    row=1, col=1
)
# Precision
fig.add_trace(
    go.Scatter(x=history["epoch"], y=history["val_precision"],
               mode="lines+markers", name="Val Precision"),
    row=1, col=2
)
# Recall
fig.add_trace(
    go.Scatter(x=history["epoch"], y=history["val_recall"],
               mode="lines+markers", name="Val Recall"),
    row=2, col=1
)
# F1
fig.add_trace(
    go.Scatter(x=history["epoch"], y=history["val_f1"],
               mode="lines+markers", name="Val F1"),
    row=2, col=2
)

fig.update_layout(
    title="Validation Metrics per Epoch",
    xaxis_title="Epoch",
    yaxis_title="Score",
    height=800, width=1000,
    template="plotly_dark",
    showlegend=True
)

# 1️⃣ Save model & tokenizer to project-root/api/saved_model
base_dir = os.path.dirname(__file__)  # e.g. .../project-root/src/cine_bert
project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))  # .../project-root

save_dir = os.path.join(project_root, "api", "saved_model")
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)
print(f"Saved model and tokenizer to: {save_dir}")

# 2️⃣ Save validation metric plot HTML to project-root/api/saved_model/metrics_plot_<timestamp>/
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metric_subdir = os.path.join(save_dir, f"metrics_plot_{timestamp}")
os.makedirs(metric_subdir, exist_ok=True)
html_path = os.path.join(metric_subdir, "validation_metrics.html")
fig.write_html(html_path)
print(f"Saved validation metrics plot to: {html_path}")
