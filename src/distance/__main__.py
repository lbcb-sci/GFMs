import numpy as np
import argparse
import edlib
import time
import multiprocessing as mp
from multiprocessing import shared_memory
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
    return pairwise_distances(embeddings, metric='cosine', n_jobs=-1)

def compute_dfunc(labels: np.array) -> np.array:
    return pairwise_distances(labels, metric='euclidean', n_jobs=-1)

def levenshtein(a, b):
    '''Normalized levenshtein distance.'''
    # lev dist is at most the length of the longer string
    maxlen = max(len(a), len(b))
    return edlib.align(a, b)['editDistance'] / maxlen

def compute_dseq(sequences: list[str], chunk_size) -> np.array:

    def compute_chunk(memname: str, shape: tuple, lo: int, hi: int):
        memory = shared_memory.SharedMemory(name=memname)
        result = np.ndarray(shape, dtype=float, buffer=memory.buf)

        for i, seq1 in enumerate(sequences[lo:hi]):
            for j, seq2 in enumerate(sequences):
                distance = levenshtein(seq1, seq2)
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

chunk_size = 100

def main():
    logger = get_logger('distance')

    args = parse_args()
    path = get_raw_data_folder() / args.path

    logger.info(f' loading data at {path}...')
    embeddings, sequences, labels = get_data(path)

    N = len(labels)
    assert N % chunk_size == 0

    logger.info(f' number of elements = {N}')

    logger.info(' computing embeddings distance matrix...')
    start = time.time()
    dmat_emb  = compute_demb(embeddings[list(embeddings.keys())[0]])
    end = time.time()
    logger.info(f' done in {end-start:.2f}seconds.')

    logger.info(' computing function distance matrix...')
    start = time.time()
    dmat_func = compute_dfunc(labels.reshape(-1, 1)) 
    end = time.time()
    logger.info(f' done in {end-start:.2f}seconds.')

    logger.info(f' computing sequence distance matrix using {N // chunk_size} processes...')
    start = time.time()
    dmat_seq = compute_dseq(sequences, chunk_size=chunk_size)
    end = time.time()
    logger.info(f' done in {end-start:.2f}seconds.')

    destination = get_dist_data_folder() / args.path
    logger.info(f' saving data at {destination}...')

    data = {
        'dist_emb':  dmat_emb,
        'dist_func': dmat_func,
        'dist_seq':  dmat_seq,
    }

    np.save(destination, data, True)

    logger.info(f' done.')

    print(dmat_emb)
    print(dmat_seq)
    print(dmat_func)

if __name__ == '__main__': main()