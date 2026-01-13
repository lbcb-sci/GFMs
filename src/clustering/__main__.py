import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from src.common import (
    get_logger,
    get_dist_data_path, 
    get_raw_data_path, 
    get_plots_path,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--clusters', required=True, type=int)
    return parser.parse_args()

def get_distance_matrices(data, check: bool = False):
    demb  = squareform(data['dist_emb'].astype(float),  checks=check)
    dseq  = squareform(data['dist_seq'].astype(float),  checks=check)
    dfunc = squareform(data['dist_func'].astype(float), checks=check)
    return demb, dseq, dfunc

def get_distance_data():
    args = parse_args()
    path = get_dist_data_path() / args.path
    data = np.load(path, allow_pickle=True)[()]
    return data

def get_emb_data():
    args = parse_args()
    path = get_raw_data_path() / args.path
    data = np.load(path, allow_pickle=True)[()]
    return data

def get_clusters(
        dmat,
        nclusters: int,
    ):
    linked = linkage(
        dmat, 
        method='ward',
        metric='euclidean',
    )
    clusters = fcluster(
        linked, 
        t=nclusters, 
        criterion='maxclust',
    )
    return clusters

def main():
    plt.style.use('bmh')
    args = parse_args()

    logger = get_logger('clustering')
    logger.info(f' args: {args}')

    nclusters = args.clusters

    dist_data = get_distance_data()
    demb, dseq, dfunc = get_distance_matrices(dist_data)

    clusters_seq = get_clusters(dseq, nclusters=nclusters)
    clusters_emb = get_clusters(demb, nclusters=nclusters)

    emb_data = get_emb_data()
    embeddings = emb_data['embeddings_layer_last']

    labels = emb_data['labels']

    mapper = umap.UMAP(
        metric='cosine',
        low_memory=False,
    )

    logger.info(' running umap clustering...')
    embeddings = mapper.fit_transform(embeddings)
    logger.info(' umap clustering done.')

    del emb_data
    palette = np.array(["tab:blue", "tab:orange", "tab:green"])

    logger.info(' generating plot...')
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    ax[0].scatter(embeddings[:, 0], embeddings[:, 1], c=palette[clusters_seq])
    ax[0].set_title('Clustering by Sequences')

    ax[1].scatter(embeddings[:, 0], embeddings[:, 1], c=palette[clusters_emb])
    ax[1].set_title('Clustering by Embeddings')

    ax[2].scatter(embeddings[:, 0], embeddings[:, 1], c=palette[labels])
    ax[2].set_title('Clustering by Function (Labels)')

    ts = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    plot_path = get_plots_path() / f'{ts}.png'
    fig.tight_layout()
    fig.savefig(plot_path, dpi=500)
    logger.info(f' plot saved at {plot_path}.')

if __name__ == '__main__': main()
