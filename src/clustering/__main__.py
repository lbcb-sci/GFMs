import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import umap

from src.common import (
    get_logger,
    get_dist_data_folder, 
    get_raw_data_folder, 
    get_plots_folder,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    return parser.parse_args()

def get_distance_matrices(data):
    demb  = squareform(data['dist_emb'].astype(float),  checks=True)
    dseq  = squareform(data['dist_seq'].astype(float),  checks=True)
    dfunc = squareform(data['dist_func'].astype(float), checks=True)
    return demb, dseq, dfunc

def get_distance_data():
    args = parse_args()
    path = get_dist_data_folder() / args.path
    data = np.load(path, allow_pickle=True)[()]
    return data

def get_emb_data():
    args = parse_args()
    path = get_raw_data_folder() / args.path
    data = np.load(path, allow_pickle=True)[()]
    return data

def get_clusters(
        dmat,
        linkage_method: str = 'complete',
        nclusters: int = 3,
    ):
    clusters = fcluster(linkage(dmat, method=linkage_method), t=nclusters, criterion='maxclust')
    return clusters

def main():
    logger = get_logger('clustering')

    dist_data = get_distance_data()
    demb, dseq, dfunc = get_distance_matrices(dist_data)

    clusters_seq = get_clusters(dseq)
    clusters_emb = get_clusters(demb)

    emb_data = get_emb_data()
    embeddings = emb_data['embeddings_layer_last']

    labels = emb_data['labels']

    mapper = umap.UMAP(low_memory=False)

    logger.info(' running umap clustering...')
    embeddings = mapper.fit_transform(embeddings)
    logger.info(' umap clustering done.')

    del emb_data

    logger.info(' generating plot...')
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    ax[0].scatter(embeddings[:, 0], embeddings[:, 1], c=clusters_seq)
    ax[0].set_title('By Sequences')

    ax[1].scatter(embeddings[:, 0], embeddings[:, 1], c=clusters_emb)
    ax[1].set_title('By Embeddings')

    ax[2].scatter(embeddings[:, 0], embeddings[:, 1], c=labels)
    ax[2].set_title('By FuncLabels')

    ts = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    plot_path = get_plots_folder() / f'{ts}.png'
    fig.savefig(plot_path, dpi=500)
    logger.info(f' plot saved at {plot_path}.')

if __name__ == '__main__': main()
