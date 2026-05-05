import torch
from pathlib import Path
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from transformers import PreTrainedTokenizer

from src.utils.paths import PATHS


class DeviceWrapper(Dataset):
    '''Dummy wrapper to move stuff to cuda if needed.'''

    def __init__(self, base, device='cuda'):
        self.base = base
        self.device = device

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in item.items()}


def get_dataset(path: str, name: str, n: int, cache_dir: str = None) -> Dataset:
    # we index from the end to get unseen-during-training samples
    dataset = load_dataset(path, name, split=f'train[-{n}:]', cache_dir=cache_dir)
    return dataset


def get_test_split(n: int, path: Path) -> Dataset:
    dataset_full = load_from_disk(path)
    dataset_size = len(dataset_full)
    return dataset_full.select(range(dataset_size-n, dataset_size)) 


def get_dataset_wiki(n: int, preprocessed: bool = True):
    if preprocessed:
        path = PATHS['wiki_dataset']
        dataset_test = get_test_split(n, path)
    else:
        cache_dir = PATHS['cache_dir']
        dataset_test = get_dataset('wikimedia/wikipedia', name='20231101.en', n=n)
    return dataset_test


def get_dataset_opengenome(n: int):
    path = PATHS['og2_dataset']
    return get_test_split(n, path)


def get_dataset_ensembl(n: int):
    path = PATHS['ensembl_dataset']
    return get_test_split(n, path)


def get_dataset_ncrna(n: int):
    path = PATHS['ncrna_dataset']
    return get_test_split(n, path)


def get_dna_dataset(type: str, n: int):
    if type == 'OG2':
        return get_dataset_opengenome(n)
    elif type == 'cDNA':
        return get_dataset_ensembl(n)
    elif type == 'ncRNA':
        return get_dataset_ncrna(n)
    else:
        raise ValueError(f'unknown DNA dataset type: {type}')


# def get_wikipedia(n: int) -> Dataset:
#     return get_dataset('wikimedia/wikipedia', name='20231101.en', n=n)


# def get_opengenome(n: int) -> Dataset:
#     # workaround to get a single dataset
#     ds = concatenate_datasets(list(load_dataset('mrochk/opengenome-clean').values()))
#     return Dataset.from_dict({'text': ds['text'][:n]})


def mlm_preprocess(batch, tokenizer: PreTrainedTokenizer, mask_prob: float):
    texts = batch['text']
    tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    input_ids = tokenized['input_ids']

    mask_labels = input_ids.clone()

    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_arr = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)

    tokenized['input_ids'][mask_arr] = tokenizer.mask_token_id
    mask_labels[~mask_arr] = -100
    tokenized['labels'] = mask_labels

    return tokenized
