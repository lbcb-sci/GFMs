import time
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances

import multiprocessing as mp
from multiprocessing import shared_memory

from src.common import get_logger, get_raw_data_folder, get_dist_data_folder
from .metrics import markov_distance

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--chunk_size', required=True, type=int)
    parser.add_argument('--kmer_markov', required=True, type=int)
    return parser.parse_args()

def get_data(path):
    data = np.load(path, allow_pickle=True)[()]
    sequences_raw = data['sequences_raw']
    sequences_tok = data['sequences_tokenized']
    labels = data['labels']
    data.pop('sequences_raw')
    data.pop('sequences_tokenized')
    data.pop('labels')
    embeddings = data
    return embeddings, sequences_raw, sequences_tok, labels

def compute_dseq(metric, sequences: list[str], chunk_size) -> np.array:

    def compute_chunk(memname: str, shape: tuple, lo: int, hi: int):
        memory = shared_memory.SharedMemory(name=memname)
        result = np.ndarray(shape, dtype=float, buffer=memory.buf)

        for i, seq1 in enumerate(sequences[lo:hi]):
            for j, seq2 in enumerate(sequences):
                distance = metric(seq1, seq2)
                result[lo+i, j] = distance

        memory.close()

    N = len(sequences)
    shape = (N, N)
    assert N % chunk_size == 0

    memory = shared_memory.SharedMemory(create=True, size=np.zeros(shape, dtype=float).nbytes)
    result = np.ndarray(shape, dtype=float, buffer=memory.buf)

    processes: list[mp.Process] = []

    for idx in range(0, N, chunk_size):
        process = mp.Process(
            target=compute_chunk, 
            args=(memory.name, shape, idx, idx+chunk_size),
        )
        process.start()
        processes.append(process)

    for p in processes: p.join()
    result = result.copy()

    memory.close()
    memory.unlink()

    return result / result.max()

def compute_demb(embeddings: np.array) -> np.array:
    return pairwise_distances(embeddings, metric='cosine', n_jobs=-1)

def compute_dfunc(labels: np.array) -> np.array:
    return pairwise_distances(labels, metric='euclidean', n_jobs=-1)

def main():
    logger = get_logger('distance')

    args = parse_args()
    logger.info(f' args: {args}')

    kmer_markov = args.kmer_markov
    chunk_size = args.chunk_size

    path = get_raw_data_folder() / args.path

    logger.info(f' loading data at {path}...')
    embeddings, sequences, sequences_tok, labels = get_data(path)

    N = len(labels)
    assert N % chunk_size == 0

    logger.info(f' number of elements = {N}')

    logger.info(' computing embeddings distance matrix...')
    start = time.time()
    dmat_emb = compute_demb(embeddings[list(embeddings.keys())[0]])
    end = time.time()
    logger.info(f' done in {end-start:.2f}s.')

    logger.info(' computing function distance matrix...')
    start = time.time()
    dmat_func = compute_dfunc(labels.reshape(-1, 1)) 
    end = time.time()
    logger.info(f' done in {end-start:.2f}s.')

    logger.info(f' computing sequence distance matrix using {N // chunk_size} processes...')
    start = time.time()
    dmat_seq = compute_dseq(lambda a, b: markov_distance(a, b, kmer_markov), sequences, chunk_size=chunk_size)
    end = time.time()
    logger.info(f' done in {end-start:.2f}s.')

    for i in range(dmat_emb.shape[0]):
        dmat_emb[i, i] = 0.0
        dmat_seq[i, i] = 0.0
        dmat_func[i, i] = 0.0

    destination = get_dist_data_folder() / args.path
    logger.info(f' saving data at {destination}...')

    data = {
        'dist_emb':  dmat_emb,
        'dist_func': dmat_func,
        'dist_seq':  dmat_seq,
    }

    np.save(destination, data, True)

    logger.info(f' done.')

if __name__ == '__main__': main()