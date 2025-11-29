"""
Script to train the Diffusion model.
"""
import sys
import os
import argparse
import torch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformers import BertTokenizer # Kept BertTokenizer
from torch.optim import AdamW # Changed to import AdamW from torch.optim
# from torch.optim.lr_scheduler import StepLR # Example scheduler - COMMENTED OUT

from LatentDiffusionTransformerCode.utils.config import load_config
from LatentDiffusionTransformerCode.utils.logger import setup_logger
from LatentDiffusionTransformerCode.models.diffusion_model import DiffusionModel
from LatentDiffusionTransformerCode.training.diffusion_trainer import DiffusionTrainer
from LatentDiffusionTransformerCode.data.dataset_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Train a Diffusion model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the diffusion config file.")
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config(args.config)
    logger = setup_logger("DiffusionTraining")
    logger.info(f"Configuration loaded from {args.config}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 2. Initialize Tokenizer
    # Make sure to use the same tokenizer that generated your vocab_size and special tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') # Changed tokenizer
    config['model_params']['vocab_size'] = tokenizer.vocab_size
    config['model_params']['mask_token_id'] = tokenizer.mask_token_id
    config['model_params']['pad_token_id'] = tokenizer.pad_token_id
    logger.info(f"Tokenizer initialized: {tokenizer.name_or_path}, Vocab Size: {tokenizer.vocab_size}")

    # 3. Load Data
    logger.info(f"Loading training data from {config['data']['path']}")
    train_loader = load_dataset(
        dataset_name_or_path=config['data']['path'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        split='train'
    )
    val_loader = load_dataset(
        dataset_name_or_path=config['data']['path'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        split='val'
    )
    logger.info(f"Train data loaded: {len(train_loader.dataset)} samples")
    logger.info(f"Validation data loaded: {len(val_loader.dataset)} samples")

    # 4. Instantiate Model, Optimizer, and Scheduler
    model = DiffusionModel(config)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = None # Disabled scheduler

    logger.info("Model, Optimizer, and Scheduler initialized.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 5. Instantiate Trainer and Run Training Loop
    trainer = DiffusionTrainer(model, optimizer, scheduler, train_loader, val_loader, config)
    logger.info("Starting training...")

    for epoch in range(config['training']['num_epochs']):
        train_loss = trainer.train_one_epoch(epoch)
        val_loss = trainer.evaluate(epoch)
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save a checkpoint after every epoch
        trainer.save_checkpoint(epoch)

    logger.info("Training finished.")

if __name__ == '__main__':
    main()