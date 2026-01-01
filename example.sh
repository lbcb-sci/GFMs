#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate emb

TASK=human_enhancers_cohn
VERSION=v2-100m-multi-species
FILEPATH=$VERSION\_$TASK.npy

echo $FILEPATH

python -m src.embeddings.bio \
    --task $TASK \
    --version $VERSION \
    --batch_size 10 \
    --limit 100 \
    --layer last

python -m src.distance.bio --path $FILEPATH 

python -m src.clustering --path $FILEPATH 
