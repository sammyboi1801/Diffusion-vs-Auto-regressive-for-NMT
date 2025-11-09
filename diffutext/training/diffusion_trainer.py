"""
Training logic for the Diffusion model.
"""
import torch
import torch.nn as nn
from .trainer_base import BaseTrainer

class DiffusionTrainer(BaseTrainer):
    """
    Trainer for the Diffusion model.
    """
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config):
        super().__init__(model, optimizer, scheduler, train_loader, val_loader, config)
        raise NotImplementedError("DiffusionTrainer initialization not implemented.")

    def train_one_epoch(self, epoch: int):
        """
        Performs one training epoch for the Diffusion model.
        The goal is to train the model to predict the noise that was added to a latent vector.
        """
        raise NotImplementedError("DiffusionTrainer train_one_epoch not implemented.")

    def evaluate(self, epoch: int):
        """
        Evaluates the Diffusion model. This could involve checking the loss on a validation set of latents.
        """
        raise NotImplementedError("DiffusionTrainer evaluation not implemented.")
