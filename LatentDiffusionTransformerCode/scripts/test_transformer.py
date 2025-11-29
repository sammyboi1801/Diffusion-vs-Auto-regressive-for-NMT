import torch
import os
import sys

# Add project root to path so imports work
sys.path.append(os.getcwd())

from models.transformer_model import TransformerModel
from inference.transformer_inference import TransformerInference
from data.tokenizer_utils import get_tokenizer

def main():
    checkpoint_path = "checkpoints/transformer_epoch_5.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {checkpoint_path} on {device}...")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found.")
        return

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get('config', {})
    mp = config_dict.get('model_params', {})

    # Load Tokenizer
    tokenizer = get_tokenizer(mp.get('tokenizer_name', 'bert-base-multilingual-cased'))

    # Initialize Model
    model = TransformerModel(
        vocab_size=mp.get('vocab_size', tokenizer.vocab_size),
        d_model=mp.get('d_model', 128),
        nhead=mp.get('nhead', 4),
        num_encoder_layers=mp.get('num_encoder_layers', 4),
        num_decoder_layers=mp.get('num_decoder_layers', 4),
        dim_feedforward=mp.get('dim_feedforward', 512),
        dropout=mp.get('dropout', 0.1)
    ).to(device)

    # Load Weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model Loaded!")

    # Inference Wrapper
    inference_engine = TransformerInference(model, tokenizer, config=None)

    print("\n--- English to French Translator ---")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            text = input("Enter English: ")
            if text.lower() == 'exit': break
            
            # Generate
            translated = inference_engine.generate(text, max_length=32)
            print(f"French: {translated}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()