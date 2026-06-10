import math 
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)  # Create a matrix of shape (max_len, d_model) to hold positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Generat eposition indices and reshape to (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Compute the scaling factor for the sine and cosine functions

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Add a batch dimension to the positional encoding matrix

        self.register_buffer('pe', pe)  # Buffer

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    
class MultiTaskTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead=4, 
                 num_encoder_layer=4, num_decoder_layer=4, pad_idx=0, tie_embeddings=False):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Token embedding layer and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model)  

        # Transformer encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layer)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layer)

        #self.classifier = nn.Linear(d_model, num_classes)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Optionally tie the output projection weights to the embedding weights
        if tie_embeddings:
            self.out_proj.weight = self.embedding.weight

    def forward_encoder(self, src_ids, src_key_padding_mask=None):
        x = self.embedding(src_ids) * math.sqrt(self.d_model)   # Embed source tokens and scale by sqrt(d_model)
        x = self.pos_enc(x) # Add positional encoding
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def summarizer(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # Encode source sequence
        memory = self.forward_encoder(src_ids, src_key_padding_mask=src_key_padding_mask)

        # Embed target sequence and add positional encoding
        tgt_emb = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)

        decoded = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=src_key_padding_mask)
        logits = self.out_proj(decoded) # Project decoder output to vocabulary size 
        return logits