import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from src.utils import get_data_folder

def get_data(path):
    data = np.load(path, allow_pickle=True)
    embeddings = data[()]['layer_22']
    sequences = data[()]['sequences']
    labels = data[()]['labels']
    return embeddings, sequences, labels

def main():
    path = get_data_folder() / 'v2-100m-multi-species_human_nontata_promoters.npy'
    embeddings, sequences, labels = get_data(path)

    dmat_emb  = cosine_distances(embeddings)
    dmat_func = (labels[:, None] != labels[None, :]).astype(float) 
    
if __name__ == '__main__': main()