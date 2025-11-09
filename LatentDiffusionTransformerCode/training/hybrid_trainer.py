"""
Custom trainer for the combined Hybrid (DiffuText) model.
"""
import torch
import torch.nn as nn
from .trainer_base import BaseTrainer

class HybridTrainer(BaseTrainer):
    """
    Trainer for the Hybrid (DiffuText) model.
    This trainer manages the two-stage training process:
    1. Train the Diffusion model on latent representations.
    2. Train the Transformer decoder to generate text from the diffused latents.
    """
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config):
        super().__init__(model, optimizer, scheduler, train_loader, val_loader, config)
        raise NotImplementedError(f"HybridTrainer initialization not implemented for stage '{self.training_stage}'.")

    def train_one_epoch(self, epoch: int):
        """
        Performs one training epoch for the Hybrid model.
        The behavior changes based on the `training_stage` config.
        """
        raise NotImplementedError("HybridTrainer one epoch not implemented.")

    def train_diffusion_epoch(self, epoch: int):
        """
        Training logic for the diffusion component of the hybrid model.
        """
        raise NotImplementedError("HybridTrainer diffusion epoch training not implemented.")

    def train_decoder_epoch(self, epoch: int):
        """
        Training logic for the decoder component of the hybrid model.
        The diffusion model's weights are frozen, and it's used to generate latents.
        """
        raise NotImplementedError("HybridTrainer decoder epoch training not implemented.")


    def evaluate(self, epoch: int):
        """
        Evaluates the hybrid model. The evaluation metric depends on the training stage.
        """
        raise NotImplementedError("HybridTrainer evaluation not implemented.")
