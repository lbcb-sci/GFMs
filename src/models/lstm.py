import torch
from torch import Tensor
from torch import nn

class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embeddings(tokens)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits
