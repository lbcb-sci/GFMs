import torch 
import numpy as np
from torch import Tensor
from functools import partial
from ckatorch.core import cka_base
from torch.nn import functional as F
from typing import Callable, Iterable
from scipy.spatial.distance import jensenshannon 

def cosine_similarity(A: Tensor, B: Tensor) -> Tensor:
    '''Compute all-tokens cosine similarity between two embeddings matrices.'''

    return F.normalize(A) @ F.normalize(B).T

def jaccard(A: list, B: list):
    '''Jaccard index = inter / union of 2 sets.'''

    setA, setB = set(A), set(B)
    inter, union = len(setA & setB), len(setA | setB)
    return inter / union if union > 0 else 0.0

def centered_kernel_alignment(A: Tensor, B: Tensor, kernel: str):
    '''Wrapper for cka_base from `ckatorch`.'''

    assert kernel in ['linear', 'rbf']
    # TODO: why do they have to be mapped to cpu for cka_base to work?
    return cka_base(A.cpu(), B.cpu(), kernel=kernel)

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

    if isinstance(data, dict): data = list(data.values())

    results = []
    for i, a in enumerate(data):
        for b in data[i+1:]: results.append(metric(a, b))
    return torch.as_tensor(results)

def cka(embeddings: Tensor, kernel: str): 
    '''Compute pairwise Centered Kernel Alignment.'''

    return meanstd(compute_pairwise(
        embeddings, 
        metric=partial(centered_kernel_alignment, kernel=kernel),
    ))

def topk(embeddings: Tensor, k: int): 
    '''Compute pairwise Top-K Overlap.'''

    return meanstd(compute_pairwise(
        embeddings, 
        metric=partial(topk_overlap, k=k),
    ))

def std_per_token(cosims: Tensor) -> True:
    '''Return mean standard deviation of cosine similarities across models, per token.'''

    assert cosims.dim() == 3
    return cosims.std(dim=0).mean().item(), torch.nan

def meanstd(values: Iterable) -> tuple[float, float]:
    '''Wrapper for mean and std.'''

    if isinstance(values, Tensor):
        return torch.mean(values).item(), torch.std(values, unbiased=False).item()
    
    return np.mean(values), np.std(values)

def jensen_shannon(probs: Tensor) -> tuple[float, float]:
    '''Pairwise Jensen-Shannon distance.'''

    return meanstd(compute_pairwise(probs.cpu(), partial(jensenshannon, base=2)))
