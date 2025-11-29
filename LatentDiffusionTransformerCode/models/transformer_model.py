"""
Baseline Transformer implementation.
"""
import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        """
        Loads a model from a saved checkpoint.
        """
        # Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Instantiate the model using the config saved in the checkpoint
        # Assuming the config is saved under keys matching __init__ args
        config = checkpoint.get('config', {})
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class PositionalEncoding(nn.Module):
    """
    Helper class to inject positional information into the embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=150000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of [max_len, d_model] representing positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(BaseModel):
    """
    A standard Transformer model for text generation/translation.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 1. Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Core Transformer (Encoder + Decoder)
        # batch_first=True ensures input format is [batch, seq_len, features]
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Output Projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Save config for checkpointing
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout
        }

    def forward(self, src, tgt):
        """
        Forward pass for the Transformer model.

        Args:
            src: Source token indices [batch, src_len]
            tgt: Target token indices [batch, tgt_len]
        """
        # 1. Create Masks
        # Target causal mask (prevents looking at future tokens in decoder)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # Padding masks (ignore 0 padding tokens)
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)

        # 2. Embeddings + Positional Encoding
        # Scale embeddings by sqrt(d_model) as per "Attention is All You Need" paper
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 3. Transformer Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask 
        )
        
        # 4. Output Logits
        return self.fc_out(output)