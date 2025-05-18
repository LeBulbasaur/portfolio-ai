import torch
from torch import nn
import lightning as L
from transformers import AutoTokenizer, AutoModel

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

class CVJobMatchingSystem:
    def __init__(self, model_path="model/cv_job_matching_model.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = CVJobMatchingModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def compare_cv_with_job(self, cv_text):
        job_offer = "We are looking for a JavaScript and Python developer with experience in machine learning and data analysis."
        encoded = self.tokenizer(
            cv_text,
            job_offer,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        with torch.no_grad():
            score = self.model(
                encoded['input_ids'],
                encoded['attention_mask']
            ).item()
        return score