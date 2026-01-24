import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

eps = 1e-8

def cosine_similarities(embeddings: Tensor) -> Tensor:
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarities = (embeddings @ embeddings.T).clamp(-1.0, 1.0)
    return similarities

def spearman(a: Tensor, b: Tensor) -> float:
    assert a.shape == b.shape

    a_rank = a.argsort().argsort().float()
    b_rank = b.argsort().argsort().float()

    a_rank = a_rank - a_rank.mean()
    b_rank = b_rank - b_rank.mean()
    num = (a_rank * b_rank).sum()
    denom = (a_rank.pow(2).sum().sqrt() * b_rank.pow(2).sum().sqrt()).clamp_min(eps)
    return (num / denom).item()

def topk_neighbor_overlap(sim_mats: list[Tensor], k: int):
    V = sim_mats[0].size(0)

    pair_overlaps = []

    for i, S1 in enumerate(sim_mats):
        _, nbrs1 = torch.topk(S1, k+1, dim=1)
        nbrs1 = nbrs1[:, 1:]

        for S2 in sim_mats[i + 1:]:
            _, nbrs2 = torch.topk(S2, k+1, dim=1)
            nbrs2 = nbrs2[:, 1:]

            # jaccard
            overlaps = []
            for t in range(V):

                n1 = set(nbrs1[t].tolist())
                n2 = set(nbrs2[t].tolist())

                inter = len(n1 & n2)
                union = len(n1 | n2)

                if union > 0: overlaps.append(inter / union)

            pair_overlaps.append(float(sum(overlaps) / len(overlaps)))

    return pair_overlaps

def local_spearman(sim_mats: list[Tensor]):
    V = sim_mats[0].size(0)

    pair_scores = []

    for i, S1 in enumerate(sim_mats):
        for S2 in sim_mats[i + 1:]:
            scores = []
            for t in range(V):

                # similarities from token t to all others, excluding self
                a = torch.cat([S1[t, :t], S1[t, t + 1:]])
                b = torch.cat([S2[t, :t], S2[t, t + 1:]])
                scores.append(spearman(a, b))

            pair_scores.append(float(sum(scores) / len(scores)))

    return pair_scores

def per_token_std(matrices: list[Tensor]) -> Tensor:
    sims = torch.stack(matrices, dim=0)
    sims = sims - sims.mean(dim=2, keepdim=True)
    std_full = sims.std(dim=0)
    per_token = std_full.mean(dim=1)
    return per_token

def relative_diff_std(glm_similarities: list, llm_similarities: list) -> float:
    l = per_token_std(llm_similarities).mean()
    g = per_token_std(glm_similarities).mean()
    return ((g - l) / l).item()

def linear_cka(emb1: Tensor, emb2: Tensor) -> float:
    Xc = emb1 - emb1.mean(dim=0, keepdim=True)
    Yc = emb2 - emb2.mean(dim=0, keepdim=True)
    K, L = Xc @ Xc.T, Yc @ Yc.T
    return (K * L).sum() / (torch.norm(K, p='fro') * torch.norm(L, p='fro')).item()

def compute_cca_corrs(X, Y, n_components=None):
    # Standardize
    X_std = StandardScaler().fit_transform(X)
    Y_std = StandardScaler().fit_transform(Y)

    if n_components is None:
        n_components = min(X_std.shape[1], Y_std.shape[1])

    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X_std, Y_std)

    corrs = []
    for k in range(n_components):
        xk = X_c[:, k]
        yk = Y_c[:, k]
        corr = np.corrcoef(xk, yk)[0, 1]
        corrs.append(corr)

    return np.array(corrs)

def compute_pairwise(metric, data):
    results = []
    for i, a in enumerate(data):
        for b in data[i+1:]: results.append(metric(a, b))
    return results

def cka(embeddings) -> list[float]:
    return compute_pairwise(linear_cka, embeddings)

def cca(embeddings):
    return compute_pairwise(compute_cca_corrs, embeddings)