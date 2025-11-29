"""
Script to run inference with a trained Diffusion model.
"""
import sys
import os
import argparse
import torch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformers import BertTokenizer

from LatentDiffusionTransformerCode.utils.config import load_config
from LatentDiffusionTransformerCode.inference.diffusion_inference import DiffusionInference


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Diffusion model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the diffusion config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument('--prompt', type=str, required=True, help="The source text prompt to translate.")
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config(args.config)
    print(f"Configuration loaded from {args.config}")

    # 2. Initialize Tokenizer
    # Ensure this matches the tokenizer used during training
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    print(f"Tokenizer initialized: {tokenizer.name_or_path}")

    # 3. Instantiate the inference engine with the trained model
    print("Initializing inference engine...")
    inference_engine = DiffusionInference(
        tokenizer=tokenizer,
        config=config,
        checkpoint_path=args.checkpoint
    )

    # 4. Generate a translation
    print("\nStarting generation...")
    translation = inference_engine.generate(args.prompt)

    print("\n--- Inference Complete ---")
    print(f"Source Text: {args.prompt}")
    print(f"Generated Translation: {translation}")
    print("--------------------------")


if __name__ == '__main__':
    main()
