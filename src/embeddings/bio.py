import numpy
import torch
import argparse
from collections import defaultdict
from src.datasets import genomic_benchmarks
from src.models import nt
from src.datasets import nt_tasks 
from src.common import (
    get_raw_data_path, 
    device, 
    get_logger,
    get_dl,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=False, default=100)
    parser.add_argument('--limit', type=int, required=False, default=None)
    parser.add_argument('--layers', type=str, required=False, default='last', choices=['last', 'all'])
    return parser.parse_args()

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    logger = get_logger('embeddings')
    logger.info(f' using device {device}')

    # parse args
    args = get_args()
    logger.info(f' args = {args}')

    task = args.task
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_version = args.version
    limit = args.limit
    layers = args.layers

    # get model and tokenizer
    model_name = nt.get_model_name(model_version) 
    model = nt.get_model_pretrained(model_name).to(device)
    tokenizer = nt.get_tokenizer(model_name)

    # get data
    logger.info(f' loading dataset {task}...')
    dataloader = get_dl(
        task=task, 
        batch_size=batch_size, 
        num_workers=num_workers,
    )
    logger.info(' loading dataset done.')

    # compute embeddings
    logger.info(f' computing embeddings...')

    data = defaultdict(list)

    for i, (sequences, labels) in enumerate(dataloader):
        embeddings = nt.get_embeddings(model, tokenizer, sequences)

        data['labels'].extend(labels)
        data['sequences_raw'].extend(sequences)
        tokenized = [tokenizer.tokenize(t) for t in sequences]
        data['sequences_tokenized'].extend(tokenized)

        match layers:
            case 'all':
                for layer, emb in enumerate(embeddings): 
                    s = f'embeddings_layer_{layer}'
                    data[s].extend(emb)
            case 'last':
                data['embeddings_layer_last'].extend(embeddings[-1])
            case _: 
                raise Exception()

        if limit is not None and (i+1) == limit: break

        print(i, end=' ', flush=True)
    print()

    for key in data.keys():
        if key == 'sequences': continue
        data[key] = numpy.array(data[key])

    # save embeddings for reuse
    data_dir = get_raw_data_path()
    filename = f'{model_version}_{task}.npy'
    final_path = data_dir / filename 

    logger.info(f'finished computing embeddings, saving file in {final_path}...')

    numpy.save(final_path, data, allow_pickle=True)
    logger.info(' saving data done.')

if __name__ == '__main__': main()
