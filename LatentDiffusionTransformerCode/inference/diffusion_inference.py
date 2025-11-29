"""
Inference logic for the LLaDA-style Diffusion model.
"""
import torch
import torch.nn.functional as F
from .inference_base import BaseInference
from LatentDiffusionTransformerCode.models.diffusion_model import DiffusionModel

class DiffusionInference(BaseInference):
    """
    Handles inference for the LLaDA-style Diffusion model.
    This involves an iterative denoising process to generate text.
    """
    def __init__(self, tokenizer, config: dict, checkpoint_path: str = None):
        """
        Initializes the DiffusionInference class.

        Args:
            tokenizer: The tokenizer for processing text.
            config (dict): A dictionary containing inference configuration.
            checkpoint_path (str, optional): Path to the trained model checkpoint. 
                                             If None, an untrained model will be initialized.
        """
        self.config = config
        
        # Instantiate the model
        model = DiffusionModel(self.config)

        # Load checkpoint if provided
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: No checkpoint provided. Using an untrained model for inference.")

        # Call the parent constructor with the instantiated model
        super().__init__(model, tokenizer, config)

        self.steps = self.config['inference']['steps']
        self.gen_length = self.config['inference']['gen_length']
        self.mask_token_id = self.config['model_params']['mask_token_id']
        print("DiffusionInference (LLaDA-style) initialized.")

    @torch.no_grad()
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a translation using the LLaDA iterative denoising strategy.

        Args:
            prompt (str): The source text (e.g., in English).
            **kwargs: Additional arguments. Accepts 'steps' and 'gen_length'
                      to override config values.

        Returns:
            The generated translation as a string.
        """
        self.model.eval()
        
        steps = kwargs.get('steps', self.steps)
        # Use gen_length from config for the target length
        trg_len = kwargs.get('gen_length', self.gen_length) 
        
        # 1. Tokenize source text
        src_tokens = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
        
        # 2. Initialize target with pure noise (all MASK tokens)
        curr_trg = torch.full((1, trg_len), self.mask_token_id, device=self.device)
        
        print(f"Source: {prompt}")

        # 3. Diffusion Loop (Iterative Denoising)
        for step in range(steps):
            # A. Concatenate source and current (noisy) target
            input_ids = torch.cat([src_tokens, curr_trg], dim=1)
            
            # B. Predict all tokens
            logits = self.model(input_ids)
            trg_logits = logits[:, src_tokens.shape[1]:, :] # Extract target part
            
            # Get probabilities and best token predictions
            probs = F.softmax(trg_logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            
            # C. Re-masking Strategy
            # Determine how many tokens to KEEP based on a linear schedule
            keep_ratio = (step + 1) / steps
            n_keep = int(trg_len * keep_ratio)
            
            if n_keep < trg_len:
                # Find the indices of the most confident predictions to keep
                sorted_probs, sorted_indices = torch.sort(max_probs, dim=1, descending=True)
                keep_indices = sorted_indices[:, :n_keep]
                
                # Create a new target sequence, starting with all masks
                new_trg = torch.full_like(curr_trg, self.mask_token_id)
                
                # Fill in the kept tokens from the current prediction
                # This uses advanced indexing to avoid a loop
                batch_indices = torch.arange(1).unsqueeze(1).to(self.device)
                new_trg[batch_indices, keep_indices] = preds[batch_indices, keep_indices]

                curr_trg = new_trg
            else:
                # Final step: no more masking, take all predictions
                curr_trg = preds
            
            # Optional: print progress
            decoded_step = self.tokenizer.decode(curr_trg[0], skip_special_tokens=True)
            print(f"Step {step+1}/{steps}: {decoded_step}")

        # 4. Decode and return the final result
        final_text = self.tokenizer.decode(curr_trg[0], skip_special_tokens=True)
        return final_text
