# Genomic Foundation Models

Analysis of Genomic Foundation Models (mostly their embeddings).

This repository aims to implement a series of experiments to critically evaluate the field of "Genomic Foundation Models". 

[This short essay](essay.pdf) explains my current view on the subject.

This project was built with `uv`, you can also run it as usual by installing the dependencies in `requirements.txt`.

After installing `uv` you should be able to directly run the commands below (from root), it will get the required dependencies automatically.

The BERT models configuration can be set in `src/instability/config.py` it defaults to the default BERT config (~90M params). 

To train $N$ models on text use:
```
uv run -m src.instability.train --type llm --n_models N
```

To train $N$ models on dna with a bpe tokenizer use:
```
uv run -m src.instability.train --type llm --tokenizer bpe --n_models N
```

To train $N$ models on dna with an overlapping k-mer tokenizer use:
```
uv run -m src.instability.train --type llm --tokenizer ovl --n_models N
```
