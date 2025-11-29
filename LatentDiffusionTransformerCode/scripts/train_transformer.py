"""
Script to train the Transformer model.
"""
import argparse
import torch
import torch.optim as optim
import os

# --- CORRECTED IMPORTS BELOW ---
from utils.config import load_config
from utils.logger import setup_logger
from models.transformer_model import TransformerModel
from training.transformer_trainer import TransformerTrainer
from data.dataset import load_dataset
from data.tokenizer_utils import get_tokenizer

class ConfigAdapter:
    """
    Wraps the dictionary config to provide helper methods 
    expected by the Trainer (like get_device).
    """
    def __init__(self, config_dict):
        self.config = config_dict
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        # Map training params for easy access if needed
        self.EPOCHS = config_dict['training']['num_epochs']
        self.BATCH_SIZE = config_dict['training']['batch_size']

    def get_device(self):
        return torch.device(self.device_str)

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the transformer config file.")
    args = parser.parse_args()

    # 1. Load configuration
    config_dict = load_config(args.config)
    wrapped_config = ConfigAdapter(config_dict) # Wrap for compatibility
    
    logger = setup_logger("TransformerTraining")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Using device: {wrapped_config.get_device()}")

    # 2. Setup Tokenizer & Data
    logger.info("Loading Tokenizer...")
    tokenizer_name = config_dict['model_params']['tokenizer_name']
    tokenizer = get_tokenizer(tokenizer_name)
    
    logger.info("Loading Dataset...")
    # Map YAML keys to function arguments
    train_loader = load_dataset(
        dataset_name_or_path=config_dict['data']['path'],
        tokenizer=tokenizer,
        batch_size=config_dict['training']['batch_size'],
        split='train'
    )
    
    val_loader = load_dataset(
        dataset_name_or_path=config_dict['data']['path'],
        tokenizer=tokenizer,
        batch_size=config_dict['evaluation']['batch_size'],
        split='val'
    )

    # 3. Initialize Model
    logger.info("Initializing Model...")
    mp = config_dict['model_params']
    model = TransformerModel(
        vocab_size=mp['vocab_size'],
        d_model=mp['d_model'],
        nhead=mp['nhead'],
        num_encoder_layers=mp['num_encoder_layers'],
        num_decoder_layers=mp['num_decoder_layers'],
        dim_feedforward=mp['dim_feedforward'],
        dropout=mp['dropout']
    ).to(wrapped_config.get_device())

    # 4. Setup Optimizer
    lr = config_dict['training']['learning_rate']
    opt_name = config_dict['training']['optimizer']
    
    if opt_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optional: Scheduler (Basic implementation)
    scheduler = None
    if config_dict['training']['scheduler'] != 'None':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # 5. Initialize Trainer
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=wrapped_config
    )

    # 6. Training Loop
    logger.info("Starting Training...")
    num_epochs = config_dict['training']['num_epochs']
    best_val_loss = float('inf')

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        # Train Step
        train_loss = trainer.train_one_epoch(epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")

        # Eval Step
        val_loss = trainer.evaluate(epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f}")

        # --- SAVE ROUTINE (Every Epoch) ---
        epoch_save_path = f"checkpoints/transformer_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'config': config_dict
        }, epoch_save_path)
        logger.info(f"Saved checkpoint: {epoch_save_path}")

        # --- SAVE BEST (Overwrites if better) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "transformer_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config_dict # Save the raw dict for future loading
            }, save_path)
            logger.info(f"--> Saved best model to {save_path}")

    logger.info("Training script completed.")

if __name__ == '__main__':
    main()