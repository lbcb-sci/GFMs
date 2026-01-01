#!/usr/bin/env bash


set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate emb


TASK=enhancers
VERSION=v2-100m-multi-species
FILEPATH=$VERSION\_$TASK.npy


python -m src.embeddings.bio \
    --task $TASK \
    --version $VERSION \
    --batch_size 10 \
    --limit 200 \
    --layer last


python -m src.distance.bio --path $FILEPATH --kmer_markov 2 --chunk_size 10


python -m src.clustering --path $FILEPATH --clusters 2 
