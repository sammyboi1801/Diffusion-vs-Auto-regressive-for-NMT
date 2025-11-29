import torch
from .inference_base import BaseInference

class TransformerInference(BaseInference):
    """
    Handles inference for the standard Transformer model.
    """
    def __init__(self, model, tokenizer, config):
        # BaseInference handles device placement safely (even if config is None)
        super().__init__(model, tokenizer, config)
        
        # We don't need to manually set self.device or .to(device) here
        # because the parent class (BaseInference) already did it!

    def generate(self, prompt: str, max_length: int = 32, **kwargs) -> str:
        """
        Generates text using the Transformer model in an autoregressive manner.
        """
        # 1. Tokenize the Source Prompt (English)
        src_encoding = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=False, 
            truncation=True
        )
        src = src_encoding['input_ids'].to(self.device)

        # 2. Initialize Decoder Input with Start-Of-Sequence (SOS/CLS) token
        # Using 101 as [CLS] for BERT-based tokenizers
        start_token = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id else 101
        tgt_indexes = [start_token]

        # 3. Autoregressive Loop
        for _ in range(max_length):
            # Prepare tensor for current target sequence
            tgt_tensor = torch.LongTensor([tgt_indexes]).to(self.device)
            
            # Forward pass (Get logits for the sequence)
            with torch.no_grad():
                # Model outputs logits: [1, seq_len, vocab_size]
                logits = self.model(src, tgt_tensor)
            
            # Get the prediction for the last token only
            last_token_logits = logits[0, -1, :]
            
            # Greedy Decode: Take the token with highest probability
            predicted_token = last_token_logits.argmax().item()
            
            # Append to sequence
            tgt_indexes.append(predicted_token)
            
            # Check for End-Of-Sequence (SEP/EOS)
            sep_token = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id else 102
            if predicted_token == sep_token:
                break

        # 4. Detokenize to String
        generated_text = self.tokenizer.decode(tgt_indexes, skip_special_tokens=True)
        return generated_text