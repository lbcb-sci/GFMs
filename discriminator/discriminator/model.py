import torch
from torch import nn, Tensor
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin

from discriminator.config import Pmain, Pmodel

class Discriminator(nn.Module, PyTorchModelHubMixin):
    '''
    Binary transformer discriminator for DNA sequences.

    Architecture: `[one-hot --> stem+pos --> transformer-layers --> head]`.

    Except input of shape [`batch_size` x `seq_len`] where `seq_len` = 8192 by default.

    Outputs vector of shape [`batch_size`]. 
    '''

    def __init__(self):
        super().__init__()

        self.stem = nn.Conv1d(
            4, Pmodel.d_model, 
            kernel_size=Pmodel.patch_size, 
            stride=Pmodel.patch_size,
        )

        seq_len = Pmain.length // Pmodel.patch_size

        self.positional_embeddings = nn.Embedding(seq_len, Pmodel.d_model)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=Pmodel.d_model,
                nhead=Pmodel.num_heads,
                dim_feedforward=Pmodel.dim_ff,
                dropout=Pmodel.dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(Pmodel.num_layers)
        ])

        self.head = nn.Linear(Pmodel.d_model, 1, bias=False)

        self.apply(self._init_weights)
        for layer in self.transformer_layers:
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.linear2.weight)

        self.cuda()

    def forward(self, tensor: Tensor) -> Tensor:
        x = self.stem(self.onehot(tensor)).transpose(1, 2)
        x = x + self.positional_embeddings(torch.arange(x.size(1), device=x.device))

        for layer in self.transformer_layers: x = layer(x)
        pooled = self.pool(x)

        return self.head(pooled).squeeze()

    @staticmethod
    def onehot(tensor: Tensor) -> Tensor:
        return F.one_hot(tensor, 4).float().transpose(-2, -1)

    @staticmethod
    def pool(tensor: Tensor): return tensor.mean(dim=1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
