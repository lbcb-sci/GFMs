import torch
from torch import Tensor
from torch.nn import functional as F

def cosine_similarities(embeddings: Tensor) -> Tensor:
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sims = (embeddings @ embeddings.T).clamp(-1, 1)
    return sims

def spearman(a: Tensor, b: Tensor) -> float:
    assert a.shape == b.shape

    a_rank = a.argsort().argsort().float()
    b_rank = b.argsort().argsort().float()

    a_rank = a_rank - a_rank.mean()
    b_rank = b_rank - b_rank.mean()
    num = (a_rank * b_rank).sum()
    denom = (a_rank.pow(2).sum().sqrt() * b_rank.pow(2).sum().sqrt()).clamp_min(1e-8)
    return (num / denom).item()

def topk_neighbor_overlap(sim_mats: list[Tensor], k: int):
    V = sim_mats[0].size(0)

    pair_overlaps = []

    for i, S1 in enumerate(sim_mats):
        # top-k neighbors per token (exclude self)
        # argsort descending
        # shape: [V, k]
        _, nbrs1 = torch.topk(S1, k + 1, dim=1)  # includes self
        # drop self idx=token itself
        # self is argmax at position 0; remove it
        nbrs1 = nbrs1[:, 1:]  # [V, k]

        for S2 in sim_mats[i + 1:]:
            _, nbrs2 = torch.topk(S2, k + 1, dim=1)
            nbrs2 = nbrs2[:, 1:]

            # Jaccard per token
            overlaps = []
            for t in range(V):
                n1 = set(nbrs1[t].tolist())
                n2 = set(nbrs2[t].tolist())
                inter = len(n1 & n2)
                union = len(n1 | n2)
                if union > 0:
                    overlaps.append(inter / union)
            pair_overlaps.append(float(sum(overlaps) / len(overlaps)))

    return pair_overlaps

def local_spearman_sim(sim_mats: list[Tensor]):
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

def relative_diff_std(glms_sims, llms_sim) -> float:
    l = per_token_std(llms_sim).mean()
    g = per_token_std(glms_sims).mean()
    return ((g - l) / l).item()
