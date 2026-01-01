# Experiments

Assume you have a downstream task in the form (seq, label) where the label relates to the *function* of the sequence.

Ideally you also have GLM embeddings from:
- Pretrained
- Random init
- Finetuned on the task sequences only (MLM)
- Finetuned on the task sequences and labels

### Experiment 1 - Distance Correlations

Build 3 distance matrices, $D_{emb}$ (cosine or other), $D_{seq}$ (any sequence dist metric) and $D_{func}$.

To build $D_{func}$, one option can be to train a small expert model on the task and extract its embeddings for each sequence. And use those to see if the GLM embeddings are somewhat correlated to the expert. 

Show that $corr(D_{emb}, D_{seq}) >> corr(D_{emb}, D_{func})$, which should be trivial.

Then use (partial) Mantel test or something similar to try to compute $corr(D_{emb}, D_{func})$ accounting for $D_{seq}$.  

There seems to be alternatives to this test so we should use as many as possible to make it robust. 

In a second step, perform the same experiment for NLP, with BERT for example. 

That is show that in a (for example) sentiment classification task, embeddings correlate to the tokens sequence but also to the underlying "meaning". At least that would be the ideal case scenario.

### Experiment 2 - Clustering

This is very similar to the first one.

The idea is to create clusters in an unsupervised manner from *sequences alone*, and show that embeddings naturally cluster by sequence similarity rather than biological function.

The advantage here is that we can have some visualization with umap for instance, plotting embeddings over seqsim clusters and over function clusters (the ones from the training task).

Then, we can run KNN/DBSCAN over embeddings only and use Normalized Mutual Information (or any other clustering metric) to get a numeric output. 

Again a parallel can be made with NLP very easily.

### Experiment 3 - Seq Features Training

An ablation study where we train a model from:
1. sequence features alone
2. embeddings alone
3. seq features + embeddings

There are many ways to get "sequence features".

And compare how much embeddings really add. If they were bio-meaningful, they would increase the performance significantly.

### Experiment 4 - Sparse Autoencoders

Something like in Evo2 paper, like attempting to reproduce their results.
