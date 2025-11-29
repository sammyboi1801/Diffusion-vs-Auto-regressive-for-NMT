from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
vocab_size = len(tok)

print(vocab_size)

#uncased 105879