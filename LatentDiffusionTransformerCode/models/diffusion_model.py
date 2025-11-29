import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Mock BaseModel for inheritance 
class BaseModel(nn.Module):
    def __init__(self): super().__init__()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()

        # Create a matrix of [max_len, d_model] representing the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        return self.pe[:, :x.size(1)]

class DiffusionModel(BaseModel):
    """
    A LLaDA-style non-autoregressive model for text generation.
    """
    def __init__(self, config: dict):
        super(DiffusionModel, self).__init__()
        self.config = config
        model_params = config['model_params']
        self.dim = model_params['dim']
        self.vocab_size = model_params['vocab_size']
        self.max_len = model_params['max_len']
        self.mask_token_id = model_params['mask_token_id']
        self.pad_token_id = model_params['pad_token_id']
        
        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.pos_embedding = SinusoidalPositionalEncoding(self.dim, self.max_len)
        
        # Transformer Encoder (Bidirectional Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim, 
            nhead=model_params['n_heads'], 
            dim_feedforward=self.dim * 4, 
            batch_first=True,
            norm_first=True # Usually stabilizes training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=model_params['n_layers'])
        self.output_head = nn.Linear(self.dim, self.vocab_size)
        
        print("DiffusionModel (LLaDA-style) initialized.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, l = input_ids.shape
        
        # Get token embeddings
        x = self.embedding(input_ids)
        
        # Add sinusoidal positional embeddings
        x = x + self.pos_embedding(x)
        
        # Padding mask (ignore pad tokens)
        # In PyTorch Transformer: True means ignore, False means attend
        padding_mask = (input_ids == self.pad_token_id)
        
        # Pass through Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Project to vocab size
        logits = self.output_head(x)
        return logits