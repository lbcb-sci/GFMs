import numpy
import argparse
from collections import defaultdict

from src.models.bio import nt
from src.datasets.bio import genomic_benchmarks, nt_tasks 

from src.common import get_raw_data_folder, device, get_logger

def get_dataloader(task: str, batch_size: int):
    if   task in genomic_benchmarks.TASKS: return genomic_benchmarks.get_dataloader(task, batch_size=batch_size)
    elif task in nt_tasks.TASKS:           return nt_tasks.get_dataloader(task, batch_size=batch_size)
    else: raise Exception()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--limit', type=int, required=False, default=None)
    parser.add_argument('--layers', type=str, required=False, default='last', choices=['last', 'all'])
    args = parser.parse_args()
    return args

def main():
    logger = get_logger('embeddings')

    logger.info(f' using device {device}')

    # parse args

    args = get_args()
    logger.info(f' args = {args}')

    task = args.task
    batch_size = args.batch_size
    model_version = args.version
    limit = args.limit
    layers = args.layers

    # get model and tokenizer

    model_name = nt.get_model_name(model_version) 
    model = nt.get_model_pretrained(model_name).to(device)
    tokenizer = nt.get_tokenizer(model_name)

    # get data

    logger.info(f' loading dataset {task}...')
    dataloader = get_dataloader(task=task, batch_size=batch_size)
    logger.info(' loading dataset done.')

    data = defaultdict(list)

    # compute embeddings

    for i, (sequences, labels) in enumerate(dataloader):
        embeddings = nt.get_embeddings(model, tokenizer, sequences)

        data['labels'].extend(labels)

        #data['sequences'].extend(sequences)

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

            case _: raise Exception()

        if limit is not None and (i+1) == limit: break

    for key in data.keys():
        if key == 'sequences': continue
        data[key] = numpy.array(data[key])

    # save embeddings for reuse

    data_dir = get_raw_data_folder()
    filename = f'{model_version}_{task}.npy'
    final_path = data_dir / filename 

    logger.info(f'finished computing embeddings, saving file in {final_path}...')

    numpy.save(final_path, data, allow_pickle=True)
    logger.info(' saving data done.')

if __name__ == '__main__': main()
