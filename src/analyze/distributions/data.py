'''Moved data related funcs here to make dynamic.py cleaner.'''

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer

class DeviceWrapper(Dataset):
    '''Dummy wrapper to move stuff to cuda if needed.'''

    def __init__(self, base, device='cuda'):
        self.base = base
        self.device = device

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in item.items()}

def get_dataset(path: str, name: str, n: int) -> Dataset:
    ## index from the end to get unseen samples
    dataset = load_dataset(path, name, split=f'train[-{n}:]')
    return dataset

def get_dataset_dna(n: int = 2000) -> Dataset:
    return get_dataset('zhangtaolab/plant-reference-genomes', name=None, n=n)

def get_dataset_text(n: int = 2000) -> Dataset:
    return get_dataset('wikimedia/wikipedia', name='20231101.en', n=n)

def mlm_preprocess(batch, tokenizer: PreTrainedTokenizer, mask_prob: float):
    texts = batch['text']
    tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    input_ids = tokenized['input_ids']

    mask_labels = input_ids.clone()

    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_arr = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)

    tokenized['input_ids'][mask_arr] = tokenizer.mask_token_id
    tokenized['labels'] = mask_labels

    return tokenized
