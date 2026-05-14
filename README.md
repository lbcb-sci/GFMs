# GFMs

Analysis of Genomic Foundation Models. This repository aims to implement a series of experiments to critically evaluate the field of "foundation" models for genomics. 

We train different ensembles each made of $N$ transformer encoder BERT models. The first ensemble is trained on English text with a byte-pair encoding tokenizer, the second on DNA sequences, also using BPE tokenization, ensuring meaningful comparison with the text models, and a third ensemble is also trained on DNA, but uses a k-mer non-overlapping tokenizer, a more widely
used tokenization scheme for genomic language models in practice.

Then, we analyze and compare text models to DNA models with respect to their output distributions, static word embeddings, contextual embeddings, and Fisher information concentration.


## Setup

This project was built with `uv`, you can also run it with your usual Python environment.

You can install `uv` here: https://docs.astral.sh/uv/getting-started/installation/

After installing `uv`, you can directly run the commands below, it will install the dependencies automatically from `pyproject.toml` and `uv.lock` file.

## Usage

There are two main modules in `src`: `train` and `analyze`. They should be called as Python modules (with `-m`).


### Configuration

The configuration of the models can be set in `src/utils/config.py`.
The paths to where you will store the dataset can be updated in `src/utils/paths.py`.

### Training

First, you will need to train the models:
```bash
uv run -m src.train --type {text, dna} --tokenizer {bpe, kmer} --data {wiki, og2, ncrna, cdna}
# all other hyperparams can be set in the utils/config.py file
```

Additionally, `--description` can be provided to distinguish the runs better.

Models are saved in `<scratch>/runs/<timestamp>_<type>_<tokenizer>_<description>/<id>` so that they can be retrieved later for analysis, and `<scratch>` directory can be updated in `src/utils/paths/py`.


For example, if you run `uv run -m src.train --type dna --tokenizer kmer --data og2`, it will create:
```bash
scratch/runs/<timestamp>_dna_kmer_og2/1
scratch/runs/<timestamp>_dna_kmer_og2/2
...
scratch/runs/<timestamp>_dna_kmer_og2/N
```
for $N$ specified in the config files ($N=5$ by default).

If you have limited resources, we recommend training less models (change $N$ to 3 or 2) and reducing their size (20M or 4M).

### Analysis

#### Distributions

We look at the distributions of BERT models over masked tokens. 

```bash
uv run -m src.analyze --type distribution --samples <NSAMPLES> --batch_size <BATCH_SIZE>
```

#### Static Word Embedings

We aim to see if models tend to agree on which tokens should be close in embedding space.

```bash
uv run -m src.analyze --type static 
```

#### Contextual Embedings

We analyze the relations between the contextual embeddings, taken from the final transformer layer.

```bash
uv run -m src.analyze --type embeddings --samples <NSAMPLES> --batch_size <BATCH_SIZE>
```

#### Fisher Information

We look at the concentration of Fisher Information with respect to each layer.

```bash
uv run -m src.analyze --type fisher --samples <NSAMPLES> --batch_size <BATCH_SIZE>
```
