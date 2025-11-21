"""
Common tokenization and detokenization logic.
"""
from typing import List
from transformers import AutoTokenizer

def get_tokenizer(model_name: str = "bert-base-multilingual-cased"):
    """
    Initializes and returns a tokenizer from Hugging Face.
    """
    # We use a multilingual model since we are doing En -> Fr
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_text(text: str, tokenizer) -> List[int]:
    """
    Tokenizes a single string of text.
    """
    # encode returns a list of integers (IDs)
    # add_special_tokens=True adds [CLS] and [SEP] which are crucial for Transformers
    return tokenizer.encode(text, add_special_tokens=True)

def detokenize_text(token_ids: List[int], tokenizer) -> str:
    """
    Detokenizes a list of token IDs back into a string.
    """
    # skip_special_tokens=True removes [CLS], [SEP], [PAD]
    return tokenizer.decode(token_ids, skip_special_tokens=True)