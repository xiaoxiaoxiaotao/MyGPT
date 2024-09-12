import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_length):
        self.text = text
        self.vocab = vocab
        self.seq_length = seq_length
        self.vocab_size = len(vocab)
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        chunk = self.text[idx:idx+self.seq_length]
        inputs = torch.tensor([self.vocab[c] for c in chunk[:-1]], dtype=torch.long)
        targets = torch.tensor([self.vocab[c] for c in chunk[1:]], dtype=torch.long)
        return inputs, targets
