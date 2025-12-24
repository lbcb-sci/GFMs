import argparse
import numpy as np
import edlib
from sklearn.metrics import pairwise_distances

from src.utils import get_logger, get_raw_data_folder, get_dist_data_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    return parser.parse_args()

def get_data(path):
    data = np.load(path, allow_pickle=True)[()]
    sequences = data['sequences']
    labels = data['labels']
    data.pop('sequences')
    data.pop('labels')
    embeddings = data
    return embeddings, sequences, labels

def compute_demb(embeddings: np.array) -> np.array:
    return pairwise_distances(embeddings, metric='cosine')

def compute_dfunc(labels: np.array) -> np.array:
    return (labels[:, None] != labels[None, :]).astype(float) 

def compute_dseq(sequences: list[str]) -> np.array:
    metric = lambda a, b: edlib.align(a, b)['editDistance']
    result = pairwise_distances(sequences, metric=metric)
    return result / result.max()

def main():
    logger = get_logger('distance')

    args = parse_args()
    path = get_raw_data_folder() / args.path

    logger.info(f' loading data at {path}...')
    embeddings, sequences, labels = get_data(path)

    logger.info(' computing embeddings distance matrix...')
    dmat_emb  = compute_demb(embeddings[list(embeddings.keys())[0]])

    logger.info(' computing function distance matrix...')
    dmat_func = compute_dfunc(labels) 

    logger.info(' computing sequences distance matrix...')
    dmat_seq = compute_dseq(sequences)

    destination = get_dist_data_folder() / args.path

    data = {
        'dist_emb': dmat_emb,
        'dist_func': dmat_func,
        'dist_seq': dmat_seq,
    }

    logger.info(f' saving data at {destination}...')
    np.save(destination, data, True)

    logger.info(f' done.')

if __name__ == '__main__': main()