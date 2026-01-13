import itertools
import numpy as np
import torch
from torch import nn

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
                kmer = sequence[idx:idx+self.kmer]
                next = sequence[idx+self.kmer]
                if kmer not in self.kmer2idx or next not in self.vocab:
                    continue
                row = self.kmer2idx[kmer]
                col = self.vocab.index(next)
                counts[row, col] += 1

        if smoothing:
            tmat = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + len(self.vocab))
        else:
            row_sums = counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            tmat = counts / row_sums

        self.tmat = tmat

    def ll(self, sequence: str) -> float:
        sequence = sequence.upper()
        result = 0.0
        n_transitions = 0

        for idx in range(len(sequence) - self.kmer):
            kmer = sequence[idx:idx + self.kmer]
            next = sequence[idx + self.kmer]
            if kmer not in self.kmer2idx or next not in self.vocab:
                continue
            row = self.kmer2idx[kmer]
            col = self.vocab.index(next)
            result += np.log(self.tmat[row, col])
            n_transitions += 1

        if n_transitions == 0: return float("-inf")
        return result / n_transitions