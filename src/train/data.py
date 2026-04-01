from datasets import load_dataset

from src.utils.paths import PATHS


def get_dataset(dataset: str, name: str, cache_dir: str, train_size: int, eval_size: int):
    split_train = f'train[:{train_size}]'
    split_eval  = f'train[{train_size}:{train_size + eval_size}]'

    dataset_train = load_dataset(dataset, name, split=split_train, cache_dir=cache_dir)
    dataset_eval  = load_dataset(dataset, name, split=split_eval, cache_dir=cache_dir)

    return dataset_train, dataset_eval


def get_dataset_text(train_size: int, eval_size: int):
    cache_dir = PATHS['cache_dir']  # '/mnt/sod2-project/csb4/wgs/lovro/huggingface'
    return get_dataset('wikimedia/wikipedia', '20231101.en', cache_dir, train_size, eval_size)


# def get_dataset_dna_plant(train_size: int, eval_size: int):
#     # TODO: Obsolete, swtiched to OpenGenome2
#     cache_dir = PATHS['cache_dir']  # '/mnt/sod2-project/csb4/wgs/lovro/huggingface'
#     return get_dataset('zhangtaolab/plant-reference-genomes', None, cache_dir, train_size, eval_size)


# def get_dataset_dna(train_size: int, eval_size: int):
#     # TODO: Obsolete, replaced by loading preprocessed dataset from disk. Keeping the function for reference.
#     ds_path = '/mnt/sod2-project/csb4/wgs/lovro/huggingface/opengenome2_subset/json/pretraining_or_both_phases/eukaryotic_genic_windows'
#     cache_dir = '/mnt/sod2-project/csb4/wgs/lovro/huggingface/opengenome2_subset'
#     return get_dataset(ds_path, None, cache_dir, train_size, eval_size)
