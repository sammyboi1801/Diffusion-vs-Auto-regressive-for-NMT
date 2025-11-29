"""
Base trainer class.
"""
import torch
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    Handles common setup like model, optimizer, and device placement.
    """
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Helper to get device safely
        if hasattr(config, 'get_device'):
            self.device = config.get_device()
        else:
            # Fallback if config doesn't have the method
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Move model to the correct device immediately upon initialization
        self.model.to(self.device)

    @abstractmethod
    def train_one_epoch(self, epoch: int):
        """
        Train the model for one epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, epoch: int):
        """
        Evaluate the model on the validation set.
        """
        raise NotImplementedError