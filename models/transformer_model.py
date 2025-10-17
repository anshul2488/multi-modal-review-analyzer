import torch
import torch.nn as nn

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x)
        x = self.output_fc(x.mean(dim=1))  # global pooling over sequence
        return x
