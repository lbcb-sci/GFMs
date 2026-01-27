'''Moved data related funcs here to make dynamic.py cleaner.'''

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class CudaWrapper(Dataset):
    '''Dummy wrapper to move stuff to cuda.'''
    def __init__(self, base, device='cuda'):
        self.base = base
        self.device = device

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in item.items()}

def get_dataset(path: str, name: str, n: int) -> Dataset:
    ## index from the end to get unseen samples
    #return load_dataset(path, name, split=f'train[-{n}:]')
    dataset = load_dataset(path, name, split=f'train[-{n}:]')
    return dataset

def get_dataset_dna(n: int = 2000) -> Dataset:
    return get_dataset('zhangtaolab/plant-reference-genomes', name=None, n=n)

def get_dataset_text(n: int = 2000) -> Dataset:
    return get_dataset('wikimedia/wikipedia', name='20231101.en', n=n)
