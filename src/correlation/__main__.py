import argparse
import numpy as np
from skbio.stats.distance import mantel, DistanceMatrix

from src.utils import get_dist_data_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    return parser.parse_args()

def get_distance_matrices(data):
    demb  = DistanceMatrix(data['dist_emb'].astype(float),  validate=False, condensed=True)
    dseq  = DistanceMatrix(data['dist_seq'].astype(float),  validate=False, condensed=True)
    dfunc = DistanceMatrix(data['dist_func'].astype(float), validate=False, condensed=True)
    return demb, dseq, dfunc

def main():
    args = parse_args()

    path = get_dist_data_folder() / args.path
    data = np.load(path, allow_pickle=True)[()]

    demb, dseq, dfunc = get_distance_matrices(data)

    out = mantel(demb, dfunc)
    print(out)
    out = mantel(demb, dseq)
    print(out)

if __name__ == '__main__': main()
