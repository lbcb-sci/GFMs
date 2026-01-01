# Embeddings
Analysis of Genomic Language Models embeddings.

This repository aims to implement a series of experiments to see if 
GLMs embeddings are "biologically meaningful" or not.

Those are described in [experiments.md](experiments.md).

Additional resources (especially the dowmstream tasks) can be found in [resources.md](resources.md).

### Updates:

- I realize that my argument is "we can't build large language models for DNA like we do in NLP, because of fundamental differences in data modality". So my project should instead be about performing experiments for both GLM and NLP models, and show that the former does not encode for underlying biological knowledge, while the other encodes for meaning as part of its embeddings and not just shallow sequence features. Therefore I have to design experiments that compare and NLP model to a GLM.  

#### Usage

Start by getting embeddings for a certain NT version:
```
python -m src.embeddings --version <VERSION> --task <TASK> --batch_size <BS> --limit <LIMIT>
```

Now we can compute the distance matrices:
```
python -m src.distance --path <PATH_TO_EMBEDDINGS>
```
