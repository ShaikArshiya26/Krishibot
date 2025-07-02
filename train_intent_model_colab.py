
# Install required libraries
!pip install transformers scikit-learn pandas torch

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import pickle

# ----------------------------------
# Load the Updated Dataset
# ----------------------------------
df = pd.read_csv("updated_nlp_intent_dataset.csv")  # Uses your uploaded file

# Show a quick preview
print("📊 Dataset Preview:")
print(df.head())

# ----------------------------------
# Encode Labels
# ----------------------------------
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# Save label encoder for use in Streamlit app
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# ----------------------------------
# Tokenization & Dataset Class
# ----------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=32, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

X_train, X_val, y_train, y_val = train_test_split(df["text"].tolist(), df["encoded_label"].tolist(), test_size=0.2, random_state=42)

train_dataset = IntentDataset(X_train, y_train)
val_dataset = IntentDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ----------------------------------
# BERT Model Setup
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# ----------------------------------
# Training the Model
# ----------------------------------
epochs = 3
model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

# ----------------------------------
# Save the Trained Model
# ----------------------------------
torch.save(model.state_dict(), "intent_model.pt")
print("✅ Model training complete and saved as 'intent_model.pt'")
