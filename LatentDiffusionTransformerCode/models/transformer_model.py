"""
Baseline Transformer implementation.
"""
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    Ensures that all models implement a forward pass and a method to load from a checkpoint.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the model.
        """
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        """
        Loads a model from a saved checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            An instance of the model with loaded weights.
        """
        raise NotImplementedError("Loading model from checkpoint not implemented.")

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
