import torch
from torch import Tensor
from torch import nn

class Linear(nn.Module):
    def __init__(
            self, 
            vocab_size: int,
            num_labels: int,
            embed_dim: int = 16,
        ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, num_labels)

    def forward(self, tokens: Tensor) -> Tensor:
        embeddings = self.embeddings(tokens)
        pooled = embeddings.mean(dim=1)
        logits = self.linear(pooled)
        return logits
