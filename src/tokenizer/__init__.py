import torch
import itertools

class KmerTokenizer:
    '''Dummy overlapping k-mer tokenizer.'''

    vocab = {'A', 'C', 'G', 'T'}
    unk_token = 0

    def __init__(self, kmer: int):
        self.vocab_size = 4**kmer + 1
        self.kmer = kmer
        self.kmer2id = {}

        for i, x in enumerate(itertools.product(self.vocab, repeat=kmer)):
            self.kmer2id[''.join(x)] = i+1

    def __call__(self, sequences: list[str]) -> torch.Tensor:
        result = []

        for sequence in sequences:
            tokens = []

            for idx in range(len(sequence)-self.kmer+1):

                kmer = sequence[idx:idx+self.kmer]

                if kmer not in self.kmer2id.keys(): 
                    tokens.append(self.unk_token)

                else: tokens.append(self.kmer2id[kmer])

            result.append(tokens)

        maxlen = max(map(lambda seq: len(seq), result))

        for t in result:
            t += [self.unk_token for _ in range(maxlen - len(t))]

        return torch.tensor(result, dtype=torch.long)
