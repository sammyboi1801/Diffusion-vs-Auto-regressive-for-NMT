import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer_base import BaseTrainer

class TransformerTrainer(BaseTrainer):
    """
    Trainer for the standard Transformer model (Seq2Seq).
    """
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config):
        super().__init__(model, optimizer, scheduler, train_loader, val_loader, config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is PAD token
        self.device = config.get_device()

    def train_one_epoch(self, epoch: int):
        """
        Performs one training epoch using Teacher Forcing.
        """
        self.model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Transformer]")
        
        for batch in pbar:
            # 1. Move data to device
            src = batch['src_ids'].to(self.device)
            tgt = batch['tgt_ids'].to(self.device)
            
            # 2. Prepare Inputs and Targets for Teacher Forcing
            # Input to Decoder: [SOS, w1, w2, ..., wn]
            # Target (Label):   [w1, w2, ..., wn, EOS]
            tgt_input = tgt[:, :-1] 
            tgt_label = tgt[:, 1:]

            # 3. Forward Pass
            self.optimizer.zero_grad()
            
            # The model forward() should handle the causal masking internally
            logits = self.model(src, tgt_input)
            
            # 4. Calculate Loss
            # Flatten logits: [batch * seq_len, vocab_size]
            # Flatten targets: [batch * seq_len]
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]), 
                tgt_label.reshape(-1)
            )
            
            # 5. Backward Pass
            loss.backward()
            
            # Gradient Clipping (Optional but recommended for Transformers)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self, epoch: int):
        """
        Evaluates the Transformer model on the validation set.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_label = tgt[:, 1:]
                
                logits = self.model(src, tgt_input)
                
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), 
                    tgt_label.reshape(-1)
                )
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss