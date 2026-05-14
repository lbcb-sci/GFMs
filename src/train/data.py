from pathlib import Path
from datasets import load_dataset, load_from_disk

from src.utils.paths import PATHS


def get_dataset(dataset: str, name: str, cache_dir: str, train_size: int, eval_size: int):
    split_train = f'train[:{train_size}]'
    split_eval  = f'train[{train_size}:{train_size + eval_size}]'

    dataset_train = load_dataset(dataset, name, split=split_train, cache_dir=cache_dir)
    dataset_eval  = load_dataset(dataset, name, split=split_eval, cache_dir=cache_dir)

    return dataset_train, dataset_eval


def get_train_eval_split(train_size: int, eval_size: int, path: Path):
    dataset_full = load_from_disk(path)
    dataset_train = dataset_full.select(range(train_size))
    dataset_eval  = dataset_full.select(range(train_size, train_size + eval_size))
    return dataset_train, dataset_eval


def choose_longest_seqs(dataset_train, dataset_eval, train_size, eval_size):
    dataset_train = dataset_train.map(lambda x: {'length': len(x['text'])})
    dataset_train = dataset_train.sort('length', reverse=True).select(range(train_size))
    dataset_train = dataset_train.remove_columns('length')

    dataset_eval = dataset_eval.map(lambda x: {'length': len(x['text'])})
    dataset_eval = dataset_eval.sort('length', reverse=True).select(range(eval_size))
    dataset_eval = dataset_eval.remove_columns('length')

    return dataset_train, dataset_eval


def get_dataset_wiki(train_size: int, eval_size: int, preprocessed: bool = True, choose_longest: bool = True):
    if preprocessed:
        path = PATHS['wiki_dataset']
        dataset_train, dataset_eval = get_train_eval_split(train_size, eval_size, path)
    else:
        cache_dir = PATHS['cache_dir']
        dataset_train, dataset_eval = get_dataset('wikimedia/wikipedia', '20231101.en', cache_dir, 3*train_size, 3*eval_size)
        if choose_longest:
            dataset_train, dataset_eval = choose_longest_seqs(dataset_train, dataset_eval, train_size, eval_size)
    return dataset_train, dataset_eval


def get_dataset_opengenome(train_size: int, eval_size: int):
    path = PATHS['og2_dataset']
    return get_train_eval_split(train_size, eval_size, path)


def get_dataset_ensembl(train_size: int, eval_size: int):
    path = PATHS['ensembl_dataset']
    return get_train_eval_split(train_size, eval_size, path)


def get_dataset_ncrna(train_size: int, eval_size: int):
    path = PATHS['ncrna_dataset']
    return get_train_eval_split(train_size, eval_size, path)
