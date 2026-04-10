import torch 
import numpy as np
from torch import Tensor
from functools import partial
from scipy.spatial import procrustes
from typing import Callable, Iterable
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

def cosine_similarity(A: Tensor, B: Tensor) -> Tensor:
    '''Compute all-tokens cosine similarity between two embeddings matrices.'''

    return pairwise_cosine_similarity(A, B)

def jaccard(A: list, B: list):
    '''Jaccard index = inter / union of 2 sets.'''

    setA, setB = set(A), set(B)
    inter, union = len(setA & setB), len(setA | setB)
    return inter / union if union > 0 else 0.0

def topk_overlap(cosimA: Tensor, cosimB: Tensor, k: int) -> Tensor:
    '''
    Top-k Jaccard overlap between two aligned cosine similarity matrices.
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

def kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    '''Base 2 KL-Div https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence'''

    p = p.clamp_min(1e-10); q = q.clamp_min(1e-10)
    assert (p.sum(dim=-1) - 1 < 1e-5).all() and (q.sum(dim=-1) - 1 < 1e-5).all()
    return (p * (torch.log2(p) - torch.log2(q))).sum(dim=-1)

def jensen_shannon_distance(p: Tensor, q: Tensor) -> Tensor:
    '''Jensen-Shannon Distance https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence'''

    m = (p + q) / 2
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return torch.sqrt(jsd)

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
        for b in data[i+1:]: 
            x = metric(a, b)
            results.append(x)

    return torch.as_tensor(results)

def topk(embeddings: Tensor, k: int): 
    '''Compute pairwise Top-k Overlap.'''

    metric = partial(topk_overlap, k=k)
    return meanstd(compute_pairwise(embeddings, metric))

def meanstd(values: Iterable) -> tuple[float, float]:
    '''Wrapper for mean and std.'''

    if isinstance(values, Tensor):
        return (
            torch.mean(values).item(), 
            torch.std(values, unbiased=False).item())
    return float(np.mean(values)), float(np.std(values))

def rankdata(x: Tensor) -> Tensor:
    '''Compute ranks along last dimension.'''

    tmp = x.argsort(dim=-1)
    ranks = torch.zeros_like(tmp, dtype=torch.float)

    arange = torch.arange(
        1, x.size(-1) + 1,
        device=x.device,
        dtype=torch.float
    ).expand_as(tmp)

    ranks.scatter_(-1, tmp, arange)
    return ranks

def local_spearman_correlation(A: Tensor, B: Tensor, k: int) -> Tensor:
    '''Local Spearman correlation between two similarity matrices.'''

    N = A.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=A.device)
    A = A[mask].view(N, N-1)
    B = B[mask].view(N, N-1)

    # from A
    topk_idx = torch.topk(A, k=k, dim=1, largest=True)[1]
    A, B = torch.gather(A, 1, topk_idx), torch.gather(B, 1, topk_idx)

    r1, r2 = rankdata(A), rankdata(B)
    r1 = r1 - r1.mean(dim=1, keepdim=True)
    r2 = r2 - r2.mean(dim=1, keepdim=True)

    numerator = (r1 * r2).sum(dim=1)
    denominator = torch.sqrt((r1 ** 2).sum(dim=1) * (r2 ** 2).sum(dim=1))
    spearman_A = (numerator / (denominator + 1e-8)).mean()

    # from B
    topk_idx = torch.topk(B, k=k, dim=1, largest=True)[1]
    A, B = torch.gather(A, 1, topk_idx), torch.gather(B, 1, topk_idx)

    r1, r2 = rankdata(A), rankdata(B)
    r1 = r1 - r1.mean(dim=1, keepdim=True)
    r2 = r2 - r2.mean(dim=1, keepdim=True)

    numerator = (r1 * r2).sum(dim=1)
    denominator = torch.sqrt((r1 ** 2).sum(dim=1) * (r2 ** 2).sum(dim=1))
    spearman_B = (numerator / (denominator + 1e-8)).mean()

    return (spearman_A + spearman_B) / 2

def local_spearman(cosine_similarities: Tensor, k: int) -> float:
    metric = partial(local_spearman_correlation, k=k)
    return meanstd(compute_pairwise(cosine_similarities, metric))

def get_procrustes(embeddings: Tensor) -> Tensor:
    '''Get Procrustes pairwise.'''

    results = []
    for i, a in enumerate(embeddings):
        for b in embeddings[i+1:]: results.append(procrustes(a, b))
    return results
