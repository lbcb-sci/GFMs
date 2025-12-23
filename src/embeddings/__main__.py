import numpy
import logging
import argparse

import torch
from torch.utils.data import DataLoader

from .datasets import genomic_benchmarks, nt_tasks 

from .models.nt import (
    get_tokenizer, 
    get_model_name, 
    get_embeddings,
    get_model_random,
    get_model_pretrained, 
)

from src.utils import get_data_folder

def get_dataloader(task: str, batch_size: int):
    if   task in genomic_benchmarks.TASKS: return genomic_benchmarks.get_dataloader(task, batch_size=batch_size)
    elif task in nt_tasks.TASKS:           return nt_tasks.get_dataloader(task, batch_size=batch_size)
    else: raise Exception()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='embeddings')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--limit', type=int, required=False, default=None)
    args = parser.parse_args()

    logger.info(f' args = {args}')

    task = args.task
    batch_size = args.batch_size
    model_version = args.version
    limit = args.limit

    model_name = get_model_name(model_version) 

    model = get_model_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)

    logger.info(f' loading dataset {task}...')
    dataloader = get_dataloader(task=task, batch_size=batch_size)
    logger.info(' done.')

    all_embeddings = []
    all_sequences = []
    all_labels = []

    for i, (sequences, labels) in enumerate(dataloader):
        embeddings = get_embeddings(model, tokenizer, sequences)

        all_embeddings.append(embeddings)
        all_sequences.append(sequences)
        all_labels.append(labels)

        if limit is not None and (i+1) == limit: break

    data_dir = get_data_folder()
    filename = f'{model_version}_{task}.npy'
    final_path = data_dir / filename 

    logger.info(f'finished computing embeddings, saving file in {final_path}...')

    data = {
        'embeddings': all_embeddings,
        'sequences': all_sequences,
        'labels': all_labels,
    }

    numpy.save(final_path, data, allow_pickle=True)
    logger.info(' done.')

if __name__ == '__main__': main()
