from typing import Optional
from torch.multiprocessing import cpu_count
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, load_from_disk

from discriminator.tokenizer import make_preprocess
from discriminator.config import Pdata

def select(ds: Dataset, n: Optional[int] = None) -> Dataset:
    col = ds.column_names[0]
    ds = ds.select(range(n)) if n is not None else ds
    return ds.rename_column(col, 'text') if col != 'text' else ds

def get_real(n: Optional[int] = None) -> Dataset:
    ds = concatenate_datasets(list(load_dataset('mrochk/opengenome-clean').values()))
    return select(ds, n)

def get_cdna(n: Optional[int] = None) -> Dataset:
    ds = load_from_disk('/root/ensembl_cdna_chunks')
    return select(ds, n)

def preprocess_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerFast, real: bool):
    n_proc = cpu_count()
    preprocess = make_preprocess(tokenizer)
    dataset = dataset.map(preprocess, num_proc=n_proc, remove_columns=['text'])
    dataset = dataset.add_column('real', [int(real)] * len(dataset))
    dataset.set_format('torch')
    return dataset

def load_generated_dataset() -> DatasetDict:
    dataset = load_from_disk(Pdata.save_path)
    dataset.set_format('torch')
    return dataset
