import torch
from torch import nn
import numpy as np
import itertools

class MarkovChain(nn.Module):
    def __init__(self, kmer: int):
        super().__init__()
        self.kmer = kmer
        self.vocab = ['A', 'C', 'G', 'T']

        self.kmer2idx = {}
        for i, x in enumerate(itertools.product(self.vocab, repeat=kmer)):
            self.kmer2idx[''.join(x)] = i

    def fit(self, sequences: list[str], smoothing: bool = False):
        num_states = 4 ** self.kmer
        counts = np.zeros(shape=(num_states, 4), dtype=np.float64)

        for sequence in sequences:
            sequence = sequence.upper()
            for idx in range(len(sequence) - self.kmer):
                ctx = sequence[idx:idx + self.kmer]
                nxt = sequence[idx + self.kmer]
                if ctx not in self.kmer2idx: continue # skip invalid k-mers if present
                row = self.kmer2idx[ctx]
                col = self.vocab.index(nxt)
                counts[row, col] += 1

        if smoothing:
            tmat = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + len(self.vocab))
        else:
            row_sums = counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            tmat = counts / row_sums

        self.tmat = tmat
