import torch
from torch import Tensor
from torch import nn

class LinearEmb(nn.Module):
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

class LinearKmerCount(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, tokens: Tensor) -> Tensor:
        B, L = tokens.shape
        device = tokens.device

        idx_flat = tokens + (torch.arange(B, device=device).unsqueeze(1) * self.vocab_size)
        idx_flat = idx_flat.view(-1)

        counts_flat = torch.bincount(
            idx_flat,
            minlength=B * self.vocab_size
        ).float()

        counts = counts_flat.view(B, self.vocab_size)
        logits = self.linear(counts)
        return logits
