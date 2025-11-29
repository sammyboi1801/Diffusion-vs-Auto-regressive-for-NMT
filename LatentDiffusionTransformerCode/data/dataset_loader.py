"""
Loads and preprocesses text datasets.
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Internal class to handle the specific CSV structure
class EnFrDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=32):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = str(row['en']) # Source: English
        tgt_text = str(row['fr']) # Target: French

        # Tokenize Source
        src_enc = self.tokenizer(
            src_text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )

        # Tokenize Target
        tgt_enc = self.tokenizer(
            tgt_text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "src_ids": src_enc['input_ids'].squeeze(0),
            "src_mask": src_enc['attention_mask'].squeeze(0),
            "tgt_ids": tgt_enc['input_ids'].squeeze(0),
            "tgt_mask": tgt_enc['attention_mask'].squeeze(0)
        }

def load_dataset(dataset_name_or_path: str, tokenizer, batch_size: int, split: str = 'train', max_samples=150000) -> DataLoader:
    """
    Loads the en-fr.csv, splits it, and returns a DataLoader.
    """
    # 1. Load Data
    # We limit rows to max_samples to ensure it runs on your machine
    df = pd.read_csv(dataset_name_or_path, nrows=max_samples)
    
    # Drop any bad lines
    df = df.dropna()

    # 2. Create Split (80% Train, 20% Val)
    train_size = int(0.8 * len(df))
    
    if split == 'train':
        split_df = df.iloc[:train_size]
        shuffle = True
    elif split == 'val' or split == 'test':
        split_df = df.iloc[train_size:]
        shuffle = False
    else:
        raise ValueError("Split must be 'train' or 'val'")

    # 3. Create Dataset Object
    dataset = EnFrDataset(split_df, tokenizer)

    # 4. Create DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0  # Set to 0 for compatibility on most OS
    )
    
    return loader