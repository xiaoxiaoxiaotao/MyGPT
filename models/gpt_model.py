import torch.nn as nn
import torch

class MyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_length):
        super(MyGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_length, embed_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)
        x = self.embedding(x) + self.position_embedding(positions)

        # 输入经过每一层 Transformer 解码器层
        for layer in self.transformer_layers:
            x = layer(x, x)

        logits = self.fc_out(x)
        return logits
