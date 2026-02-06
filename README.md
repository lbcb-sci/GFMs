# GFMs

Analysis of Genomic Foundation Models. This repository aims to implement a series of experiments to critically evaluate the field of "foundation" models for genomics. 

## Setup

This project was built with `uv`, you can also run it as usual by installing the dependencies in `requirements.txt`.

You can install `uv` here: https://docs.astral.sh/uv/getting-started/installation/

After installing `uv`, you can directly run the commands below, it will install the dependencies automatically from the `uv.lock` file.

## Usage

There are two main module in `src`, `train` and `analyze`. They should be called as modules (with `-m`).

### Configuration

The configuration of the models can be set in `src/utils/config.py` it defaults to the default HuggingFace BERT config (rougly ~90M params) and 5b tokens for all models. 

### Training

First, you will need to train the models:
```bash
uv run -m src.train --type {text, dna} --tokenizer {bpe, kmer} --size {4M, 20M, 90M}
# all other hyperparams can be set in the utils/config.py file
```

Models are saved in `runs/<timestamp>_<type>_<tokenizer>/<id>` so that they can be retrieved later for analysis.

For example, if you run `uv run -m src.train --type dna --tokenizer kler`, it will create:
```bash
runs/<timestamp>_dna_kmer/1
runs/<timestamp>_dna_kmer/2
...
runs/<timestamp>_dna_kmer/N
```
for $N$ specified in the config files ($N=5$ by default).

### Analysis

#### Static (Word Embedings)

We aim to see if models tend to agree on which tokens should be close in embedding space.

#### Distributions

We look at the distributions of BERT models over masked tokens of unseen sequence during training. 
