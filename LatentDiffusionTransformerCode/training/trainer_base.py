"""
Base Trainer class with shared methods for training loops, evaluation, and checkpointing.
"""
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    """
    def __init__(self, model, optimizer, scheduler, train_loader: DataLoader, val_loader: DataLoader, config: dict):
        """
        Initializes the BaseTrainer.

        Args:
            model: The model to be trained.
            optimizer: The optimizer for training.
            scheduler: The learning rate scheduler.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            config (dict): A dictionary containing training configuration.
        """
        raise NotImplementedError("BaseTrainer initialization not implemented.")

    @abstractmethod
    def train_one_epoch(self, epoch: int):
        """
        Runs a single training epoch.

        Args:
            epoch (int): The current epoch number.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, epoch: int):
        """
        Runs evaluation on the validation set.

        Args:
            epoch (int): The current epoch number.
        """
        raise NotImplementedError

    def train(self):
        """
        The main training loop.
        """
        raise NotImplementedError("Trainer function not implemented.")

    def save_checkpoint(self, epoch: int):
        """
        Saves a model checkpoint.

        Args:
            epoch (int): The current epoch number, used for naming the checkpoint.
        """
        raise NotImplementedError("Saving checkpoint not implemented.")
