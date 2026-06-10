import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PredictionModel(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_heads=4, dropout=0.3):
        super().__init__()

        # --- Temporal encoder ---
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)

        # --- Attention over all 60 LSTM output step ---
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # --- Shared trunk ---
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # --- Output Heads ---
        self.opportunity_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, 1)
        #self.sigmod = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                                  # Encodes temporal structure
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # Self-attention learns which bar are most predictive 
        out = self.attn_norm(lstm_out + attn_out)                   # Stabilises training
        out = out.mean(dim=1)                                       # Pool to a single vector 

        out = self.relu(out)
        out = self.dropout(out)

        opp = self.opportunity_head(out)
        direction = self.direction_head(out)

        return opp, direction