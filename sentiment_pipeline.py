# sentiment_pipeline.py

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

# 1. Load IMDb dataset
dataset = load_dataset("imdb")

# 2. Load tokenizer and tokenize dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

tokenized = dataset.map(tokenize, batched=True)

# 3. Load pre-trained BERT model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 4. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# 5. Prepare training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01
)

# 6. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].shuffle(seed=42).select(range(10000)),
    eval_dataset=tokenized["test"].select(range(2000)),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# 7. Train model
trainer.train()

# 8. Evaluate
metrics = trainer.evaluate()
print(metrics)

# 9. Save model
model.save_pretrained("sentiment-bert-model")
tokenizer.save_pretrained("sentiment-bert-model")

# 10. Load for inference (optional)
loaded_model = BertForSequenceClassification.from_pretrained("sentiment-bert-model")
loaded_tokenizer = BertTokenizer.from_pretrained("sentiment-bert-model")

def predict_sentiment(text):
    inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = loaded_model(**inputs).logits
    pred = torch.argmax(logits).item()
    return "Positive" if pred == 1 else "Negative"

# Sample inference
print(predict_sentiment("The movie was amazing and very entertaining!"))
