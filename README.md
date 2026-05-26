# GFMs

Analysis of Genomic Foundation Models. This repository aims to implement a series of experiments to critically evaluate the field of "foundation" models for genomics. 

We train different ensembles each made of $N$ transformer encoder BERT models. The first ensemble is trained on English text with a byte-pair encoding tokenizer, the second on DNA sequences, also using BPE tokenization, ensuring meaningful comparison with the text models, and a third ensemble is also trained on DNA, but uses a k-mer non-overlapping tokenizer, a more widely
used tokenization scheme for genomic language models in practice.

Then, we analyze and compare text models to DNA models with respect to their output distributions, static word embeddings, contextual embeddings, and Fisher information concentration.

## Setup

This project was built with `uv`, you can also run it with your usual Python environment.

You can install `uv` here: https://docs.astral.sh/uv/getting-started/installation/

After installing `uv`, you can directly run the commands below, it will install the dependencies automatically from `pyproject.toml` and `uv.lock` file.

## Data

Before training the models, you will need to download the data. The OpenGenome2 eukaryotic genic windows are available on HuggingFace, while ncRNA and cDNA data were downloaded from Ensembl. 

### OpenGenome2 data

For downloading the OpenGenome2 data from HuggingFace, you will first need `hf` - HuggingFace hub CLI. Find the installation instructions on their website: https://huggingface.co/docs/huggingface_hub/en/installation.

Once installed, run the following command to download the data:
```bash
hf download arcinstitute/opengenome2 \
  --repo-type dataset \
  --include "json/pretraining_or_both_phases/eukaryotic_genic_windows/**" \
  --local-dir opengenome2_data
```
### Ensembl data

The easiest way to download the Ensembl data is by running the `scripts/download_ensembl.py`, where you can specify with an argument whether you want to download ncRNA or cDNA. To get the datasets as in the paper, run the following command for cDNA:
```bash
python scripts/download_ensembl.py --type cdna --output-dir <output-dir> --dataset-output <dataset-output> --no-dedup
```
For ncRNA, we need to allow partial chunks due to the lack of long sequences:
```bash
python scripts/download_ensembl.py --type ncrna --output-dir <output-dir> --dataset-output <dataset-output> --chunk-size 3072 --keep-partial --min-length 2048 --no-dedup
```

In the commands above, `<output-dir>` is a directory where the raw FASTA data will be downloaded, and `<dataset-output>` is a directory where the HuggingFace dataset in the Arrow format will be stored.

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

## Discriminator

The code for training the discriminator used to reweight OpenGenome2 is located in the `discriminator` directory.

The provided scripts should be called in this order:
- `preprocess.py`
- `train.py`
- `inference.py`

This will generate a weighted dataset made of a `text` (the dna sequence) and `weight` (the "information score" derived from the discriminator logits) columns.

You can then use it to train your LM of choice on it, for example by passing the weights to a PyTorch `WeightedRandomSampler` object (see https://docs.pytorch.org/docs/2.12/data.html#torch.utils.data.WeightedRandomSampler for more information).

The discriminator architecture and different settings can be modified in `config.py` and `model.py`.  