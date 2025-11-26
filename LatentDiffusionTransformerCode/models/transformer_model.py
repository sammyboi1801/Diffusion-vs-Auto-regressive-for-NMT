"""
Baseline Transformer implementation.
"""
import torch
import math
import torch.nn as nn
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

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings,
    so that the two can be summed.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(BaseModel):
    """
    A standard Transformer model for text generation.
    This can be an encoder-decoder or a decoder-only architecture.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Initializes the Transformer model.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TransformerModel, self).__init__()
        raise NotImplementedError("TransformerModel initialization not implemented.")

    def forward(self, src, tgt):
        """
        Forward pass for the Transformer model.

        Args:
            src: The sequence to the encoder (e.g., source sentence).
            tgt: The sequence to the decoder (e.g., target sentence).

        Returns:
            Output tensor from the decoder.
        """
        raise NotImplementedError("TransformerModel forward pass not implemented.")
