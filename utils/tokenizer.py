class SimpleTokenizer:
    def __init__(self, text):
        self.vocab = {ch: idx for idx, ch in enumerate(set(text))}
        self.inv_vocab = {idx: ch for ch, idx in self.vocab.items()}
    
    def encode(self, text):
        return [self.vocab[ch] for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.inv_vocab[token] for token in tokens])
