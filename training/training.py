# pip install lightning transformers -q
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv("./datasetino resumino/dataset_random_12b.csv", header=None, names=["job_offer", "cv", "score"])
print(df.head())

df.dropna(inplace=True)
df['score'] = df['score'].astype(float) / 100.0

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class CVJobDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            row['cv'],
            row['job_offer'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(row['score'], dtype=torch.float)
        }

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_ds = CVJobDataset(train_df, tokenizer)
val_ds = CVJobDataset(val_df, tokenizer)
test_ds = CVJobDataset(test_df, tokenizer)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
test_loader = DataLoader(test_ds, batch_size=4)

class CVJobMatchingModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.MSELoss()
        self.test_losses = []

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output).squeeze()

    def training_step(self, batch, batch_idx):
        preds = self(batch['input_ids'], batch['attention_mask'])
        labels = batch['label']
        loss = self.loss_fn(preds, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(preds, batch['label'])
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        preds = self(batch['input_ids'], batch['attention_mask'])
        labels = batch['label']
        loss = self.loss_fn(preds, labels)
        self.test_losses.append(loss.detach())

        preds_list = preds.detach().cpu().numpy().tolist()
        labels_list = labels.detach().cpu().numpy().tolist()

        if not isinstance(preds_list, list):
            preds_list = [preds_list]
        if not isinstance(labels_list, list):
            labels_list = [labels_list]

        self.y_true.extend(labels_list)
        self.y_pred.extend(preds_list)

        self.log("test_loss", loss)
        return loss

    def on_test_epoch_start(self):
        self.test_losses = []
        self.y_true = []
        self.y_pred = []

    def on_test_epoch_end(self):
        y_true_scaled = [y * 100 for y in self.y_true]
        y_pred_scaled = [y * 100 for y in self.y_pred]

        mse = mean_squared_error(y_true_scaled, y_pred_scaled)
        mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
        r2 = r2_score(y_true_scaled, y_pred_scaled)

        print(f"\n📊 Test metrics (scaled to 0–100):")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        metrics = {'MSE': mse, 'MAE': mae, 'R²': r2}
        names = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(6, 4))
        bars = plt.bar(names, values, color=['skyblue', 'orange', 'green'])
        plt.title("Regression Metrics on Test Set")
        plt.ylabel("Value")
        plt.ylim(0, max(values) * 1.2)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        errors = np.array(y_true_scaled) - np.array(y_pred_scaled)
        plt.figure(figsize=(10,5))
        sns.histplot(errors, bins=30, kde=True)
        plt.title("Histogram błędów predykcji (y_true - y_pred)")
        plt.xlabel("Błąd")
        plt.ylabel("Liczba próbek")
        plt.show()

        plt.figure(figsize=(8,8))
        plt.scatter(y_true_scaled, y_pred_scaled, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel("Wartości prawdziwe")
        plt.ylabel("Wartości przewidywane")
        plt.title("Scatter plot: prawdziwe vs. przewidywane")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True)
        plt.show()

        mean_vals = (np.array(y_true_scaled) + np.array(y_pred_scaled)) / 2
        diff_vals = np.array(y_true_scaled) - np.array(y_pred_scaled)
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        plt.figure(figsize=(8,6))
        plt.scatter(mean_vals, diff_vals, alpha=0.5)
        plt.axhline(mean_diff, color='red', linestyle='--')
        plt.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--')
        plt.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--')
        plt.xlabel("Średnia (prawdziwe i przewidywane)")
        plt.ylabel("Różnica (prawdziwe - przewidywane)")
        plt.title("Bland-Altman plot")
        plt.grid(True)
        plt.show()

        def group_to_20_classes(values):
            values = np.clip(values.astype(int), 0, 100)
            return (values // 5).astype(int)

        label_20 = group_to_20_classes(np.array(y_true_scaled).round())
        pred_20 = group_to_20_classes(np.array(y_pred_scaled).round())

        cm_20 = confusion_matrix(label_20, pred_20, labels=range(20))

        plt.figure(figsize=(12,10))
        sns.heatmap(cm_20, annot=True, fmt='d', cmap='coolwarm')
        plt.title("Heatmapa błędów - grupowanie do 20 klas")
        plt.xlabel("Przewidywana klasa")
        plt.ylabel("Prawdziwa klasa")
        plt.show()


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

model = CVJobMatchingModel()
trainer = L.Trainer(max_epochs=9, accelerator='auto', devices='auto')
trainer.fit(model, train_loader, val_loader)

model_dir = os.path.join(os.path.dirname(os.getcwd()), "model")
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, "cv_job_matching_model.pt"))
trainer.test(model, test_loader)