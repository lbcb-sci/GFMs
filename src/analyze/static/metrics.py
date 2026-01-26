import torch 
from torch import Tensor
from torch.nn import functional as F
from ckatorch.core import cka_base

from functools import partial
from typing import Callable, Iterable

def cosine_similarity(A: Tensor, B: Tensor) -> Tensor:
    '''Compute all-tokens cosine similarity between two embeddings matrices.'''
    return F.normalize(A) @ F.normalize(B).T

def jaccard(A: list, B: list):
    '''Jaccard index = inter / union of 2 sets.'''
    setA, setB = set(A), set(B)
    inter, union = len(setA & setB), len(setA | setB)
    return inter / union if union > 0 else 0.0

def centered_kernel_alignment(A: Tensor, B: Tensor):
    '''Wrapper for cka_base from `ckatorch`.'''
    # TODO: why do they have to be mapped to cpu for cka_base to work?
    return cka_base(A.cpu(), B.cpu(), kernel='linear')

def topk_overlap(cosimA: Tensor, cosimB: Tensor, k: int) -> Tensor:
    '''
    Top-k Jaccard overlap between two aligned cosine similarities matrices.
    Average over all neighbor sets.
    '''
    assert tuple(cosimA.shape) == tuple(cosimB.shape)
    V = cosimA.shape[0]

    overlaps = []
    for i in range(V):

        # get the k closest tokens for each token in vocab V
        closestA = torch.topk(cosimA[i], k=k+1, largest=True, sorted=True).indices[1:]
        closestB = torch.topk(cosimB[i], k=k+1, largest=True, sorted=True).indices[1:]

        # jaccard (not efficient but k is small)
        J = jaccard(closestA.tolist(), closestB.tolist())
        overlaps.append(J)
    
    return torch.mean(torch.as_tensor(overlaps))

def compute_pairwise(data: Iterable, metric: Callable) -> Tensor:
    '''
    Compute metric `metric` over all unique pairs in `data`.

    Same as sklearn `metrics.pairwise_distances` but I didn't want to have
    to include the entire sklearn package just for that.
    '''
    assert callable(metric)
    results = []
    for i, a in enumerate(data):
        for b in data[i+1:]: results.append(metric(a, b))
    return torch.as_tensor(results)

def cka(embeddings: Tensor): 
    '''Compute pairwise Centered Kernel Alignment.'''
    return meanstd(compute_pairwise(embeddings, metric=centered_kernel_alignment))

def topk(embeddings: Tensor, k: int): 
    '''Compute pairwise Top-K Overlap.'''
    return meanstd(compute_pairwise(embeddings, metric=partial(topk_overlap, k=k)))

def meanstd(values: Iterable) -> tuple[float, float]:
    '''Wrapper for mean and std.'''
    return torch.mean(values).item(), torch.std(values).item()

def std_per_token(cosims_stacked: Tensor) -> True:
    '''Return mean standard deviation of cosine similarities across models, per token.'''
    assert cosims_stacked.dim() == 3
    return cosims_stacked.std(dim=0).mean(), 0.0
