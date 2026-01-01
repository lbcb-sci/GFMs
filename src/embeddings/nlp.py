import numpy
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import pairwise_distances
from skbio.stats.distance import mantel
import Levenshtein
import textdistance

def levenshtein(a, b):
    '''Normalized levenshtein distance.'''
    # lev dist is at most the length of the longer string
    maxlen = max(len(a), len(b))
    return Levenshtein.distance(a, b) / maxlen

def compute_dseq(sequences: list[str], chunk_size) -> np.array:

    def compute_chunk(memname: str, shape: tuple, lo: int, hi: int):
        memory = shared_memory.SharedMemory(name=memname)
        result = np.ndarray(shape, dtype=float, buffer=memory.buf)

        metric = textdistance.EntropyNCD() 

        for i, seq1 in enumerate(sequences[lo:hi]):
            for j, seq2 in enumerate(sequences):
                #distance = metric.normalized_distance(seq1, seq2)
                distance = textdistance.jaccard.normalized_distance(seq1, seq2)
                #distance = levenshtein(seq1, seq2)
                result[lo+i, j] = distance

        memory.close()

    N = len(sequences)
    shape = (N, N)
    assert N % chunk_size == 0

    memory = shared_memory.SharedMemory(create=True, size=np.zeros(shape, dtype=float).nbytes)
    result = np.ndarray(shape, dtype=float, buffer=memory.buf)

    processes: list[mp.Process] = []

    for idx in range(0, N, chunk_size):
        process = mp.Process(
            target=compute_chunk, 
            args=(memory.name, shape, idx, idx+chunk_size),
        )
        process.start()
        processes.append(process)

    for p in processes: p.join()
    result = result.copy()

    memory.close()
    memory.unlink()

    return result / result.max()

@torch.no_grad()
@torch.autograd.inference_mode()
def main():

    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased",
    )

    model = AutoModel.from_pretrained(
        "google-bert/bert-base-uncased",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    batch_size = 10

    dataset = load_dataset('stanfordnlp/imdb', split='train')
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=batch_size,
        shuffle=True, 
    )

    embeddings = []
    labels = []
    sequences = []

    for i, sample in enumerate(dataloader):
        text = sample['text']
        label = sample['label']

        tokenized = [tokenizer.tokenize(t) for t in text]

        labels.extend(label.cpu().numpy())

        tokens = tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(model.device)

        input_ids = tokens['input_ids']
        masks = tokens['attention_mask']

        sequences.extend(tokenized)

        out = model(input_ids=input_ids, attention_mask=masks)
        mask = torch.unsqueeze(masks, dim=-1).float()
        pooled_embeddings = torch.sum(mask * out.last_hidden_state, dim=1) / torch.sum(mask, dim=1)

        embeddings.extend(pooled_embeddings.detach().cpu().numpy())

        print(i+1)
        if i+1 == 1: break

    embeddings = numpy.array(embeddings, dtype=float)
    labels = numpy.array(labels, dtype=float)

    dmat_emb  = pairwise_distances(embeddings, metric='cosine')
    dmat_seq  = compute_dseq(sequences, 10)
    dmat_func = pairwise_distances(labels.reshape(-1, 1), metric='euclidean')

    print(dmat_seq)

    print()
    res = mantel(dmat_emb, dmat_func)
    print(res)
    print()
    res = mantel(dmat_emb, dmat_seq)
    print(res)

if __name__ == '__main__': main()