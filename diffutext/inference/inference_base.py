"""
Shared inference utilities and base class.
"""
from abc import ABC, abstractmethod
import torch

class BaseInference(ABC):
    """
    Abstract base class for all inference modules.
    """
    def __init__(self, model, tokenizer, config: dict):
        """
        Initializes the BaseInference class.

        Args:
            model: The pre-trained model to use for inference.
            tokenizer: The tokenizer for processing text.
            config (dict): A dictionary containing inference configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        print("Initializing BaseInference...")

    @abstractmethod
    def generate(self, prompt: str, max_length: int, **kwargs):
        """
        Generates text based on a given prompt.

        Args:
            prompt (str): The input text to start generation from.
            max_length (int): The maximum length of the generated sequence.
            **kwargs: Additional model-specific arguments.

        Returns:
            The generated text as a string.
        """
        raise NotImplementedError
