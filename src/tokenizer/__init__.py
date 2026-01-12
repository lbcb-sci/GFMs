# Simple kmer tokenizer.

import torch
import itertools

class KMERTokenizer():
    vocab = {'A', 'C', 'G', 'T'}
    unk_tokens = 0

    def __init__(self, kmer: int):
        self.vocab_size = 4**kmer + 1
        self.kmer = kmer
        self.kmer2id = {}

        for i, x in enumerate(itertools.product(self.vocab, repeat=kmer)):
            self.kmer2id[''.join(x)] = i+1

    def __call__(self, sequences):
        result = []
        for sequence in sequences:
            tokens = []
            for idx in range(len(sequence)-self.kmer+1):
                kmer = sequence[idx:idx+self.kmer]
                if kmer not in self.kmer2id.keys(): 
                    tokens.append(self.unk_tokens)
                else: tokens.append(self.kmer2id[kmer])
            result.append(tokens)

        return torch.tensor(result, dtype=torch.long)
