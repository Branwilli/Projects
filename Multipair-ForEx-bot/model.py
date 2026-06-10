import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAIRS = ['EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD']
PAIR_TO_IDX = {p: i for i, p in enumerate(PAIRS)}
N_PAIRS = len(PAIRS)

class PredictionModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_heads=4,
        dropout=0.3,
        n_pairs=N_PAIRS,
        embedding_dim=16,
    ):
        super().__init__()
 
        self.embedding_dim = embedding_dim
 
        # Pair embedding — maps pair index to a dense vector appended to
        # every timestep so the LSTM sees pair identity at each step.
        self.pair_embedding = nn.Embedding(n_pairs, embedding_dim)
 
        # LSTM input size grows by embedding_dim
        lstm_input = input_size + embedding_dim
        self.lstm  = nn.LSTM(
            lstm_input, hidden_size,
            num_layers=2, batch_first=True, dropout=dropout
        )
 
        # Self-attention over all 60 LSTM output steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
 
        # Shared trunk
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
 
        # Output heads
        self.opportunity_head = nn.Linear(hidden_size, 1)
        self.direction_head   = nn.Linear(hidden_size, 1)
 

    def forward(self, x, pair_idx):
        # Look up embedding and expand across the sequence dimension
        emb = self.pair_embedding(pair_idx)          # (batch, embedding_dim)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq, emb_dim)
 
        # Concatenate embedding onto every timestep
        x = torch.cat([x, emb], dim=-1)              # (batch, seq, features+emb)
 
        # LSTM temporal encoding
        lstm_out, _ = self.lstm(x)                   # (batch, seq, hidden)
 
        # Self-attention — learn which bars matter most
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
 
        # Residual + LayerNorm + mean pool
        out = self.attn_norm(lstm_out + attn_out)    # (batch, seq, hidden)
        out = out.mean(dim=1)                        # (batch, hidden)
 
        out = self.relu(out)
        out = self.dropout(out)
 
        opp       = self.opportunity_head(out)       # (batch, 1)
        direction = self.direction_head(out)         # (batch, 1)
 
        return opp, direction