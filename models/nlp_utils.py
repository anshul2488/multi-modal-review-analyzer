from transformers import AutoTokenizer, AutoModel
import torch

class SBERTEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def encode(self, texts, max_length=128):
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
            emb = output.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).expand(emb.size())
            emb = (emb * mask).sum(dim=1) / mask.sum(dim=1)
            return emb.cpu().numpy()
