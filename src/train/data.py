from datasets import load_dataset

def get_dataset(dataset: str, name: str, train_size: int, eval_size: int):
    split_train = f'train[:{train_size}]'
    split_eval  = f'train[{train_size}:{train_size + eval_size}]'

    dataset_train = load_dataset(dataset, name, split=split_train)
    dataset_eval  = load_dataset(dataset, name, split=split_eval)

    return dataset_train, dataset_eval

def get_dataset_text(train_size: int, eval_size: int):
    return get_dataset('wikimedia/wikipedia', '20231101.en', train_size, eval_size)

def get_dataset_dna(train_size: int, eval_size: int):
    return get_dataset('zhangtaolab/plant-reference-genomes', None, train_size, eval_size)
