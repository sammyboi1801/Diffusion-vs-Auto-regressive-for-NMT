"""
Training logic for the LLaDA-style Diffusion model.
"""
import os
import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .trainer_base import BaseTrainer

class DiffusionTrainer(BaseTrainer):
    """
    Trainer for the LLaDA-style Diffusion model.
    """
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mask_token_id = self.model.mask_token_id
        self.pad_token_id = self.model.pad_token_id
        self.vocab_size = self.model.vocab_size
        
        self.model.to(self.device)
        self.save_dir = self.config['training'].get('save_dir', 'trained_models')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("DiffusionTrainer (LLaDA-style) initialized.")

    def _calculate_loss(self, batch):
        """
        Shared logic for calculating the masked loss for seq2seq.
        """
        src_tokens = batch['src_ids'].to(self.device)
        trg_tokens = batch['tgt_ids'].to(self.device)
        
        batch_size, trg_len = trg_tokens.shape
        
        # 1. Sample a random time step 't' for each example (0 to 1)
        t = torch.rand(batch_size, device=self.device)
        
        # 2. Determine how many tokens to mask based on t (Linear Schedule)
        num_to_mask = (t * trg_len).long()
        
        # Create the noisy input and the mask matrix
        noisy_trg = trg_tokens.clone()
        mask_matrix = torch.zeros_like(trg_tokens, dtype=torch.bool)

        for i in range(batch_size):
            # Randomly select indices to mask in the target part
            # Make sure not to mask padding tokens
            non_pad_indices = (trg_tokens[i] != self.pad_token_id).nonzero(as_tuple=True)[0]
            
            # Shuffle and select a subset of non-pad tokens to mask
            perm = torch.randperm(len(non_pad_indices), device=self.device)
            num_to_mask_i = min(num_to_mask[i], len(non_pad_indices)) # Can't mask more than available tokens
            
            mask_indices_i = non_pad_indices[perm[:num_to_mask_i]]

            noisy_trg[i, mask_indices_i] = self.mask_token_id
            mask_matrix[i, mask_indices_i] = True
            
        # 3. Concatenate Source + Noisy Target
        input_ids = torch.cat([src_tokens, noisy_trg], dim=1)
        
        # 4. Forward Pass
        logits = self.model(input_ids)
        
        # 5. Extract only the logits corresponding to the target part
        trg_start_idx = src_tokens.shape[1]
        trg_logits = logits[:, trg_start_idx:, :]
        
        # 6. Calculate Loss only on the masked tokens
        loss = F.cross_entropy(
            trg_logits.reshape(-1, self.vocab_size), 
            trg_tokens.reshape(-1), 
            reduction='none'
        )
        
        flat_mask = mask_matrix.reshape(-1)
        masked_loss = (loss * flat_mask.float()).sum() / (flat_mask.sum() + 1e-6)
        
        return masked_loss

    def train_one_epoch(self, epoch: int):
        """
        Performs one training epoch for the LLaDA-style model.
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            loss = self._calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} [TRAIN] Average Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, epoch: int):
        """
        Evaluates the LLaDA-style model on the validation set.
        """
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [EVAL]")
        
        with torch.no_grad():
            for batch in progress_bar:
                loss = self._calculate_loss(batch)
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch+1} [EVAL] Average Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch: int):
        """
        Saves a model checkpoint.

        Args:
            epoch (int): The current epoch number.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"diffusion_model_{date_str}_epoch_{epoch+1}.pt"
        save_path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        
        print(f"Checkpoint saved to {save_path}")

