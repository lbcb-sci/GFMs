import torch 
from torch import Tensor
from torch.nn import functional as F
from transformers import BertForMaskedLM, BertModel

from src.common import get_models_path

def linear_cka(X: Tensor, Y: Tensor) -> float:
    assert X.shape[0] == Y.shape[0]

    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    K = Xc @ Xc.T
    L = Yc @ Yc.T

    def hs_norm(M: Tensor): return torch.norm(M, p='fro')

    hs_xy = (K * L).sum()
    hs_xx = hs_norm(K)
    hs_yy = hs_norm(L)

    cka = hs_xy / (hs_xx * hs_yy + 1e-12)
    return cka.item()

def cosine_distances(embeddings, zscore: bool = False):
    if zscore:
        mean = embeddings.mean(dim=0, keepdim=True)
        std  = embeddings.std(dim=0, unbiased=False, keepdim=True)
        embeddings  = (embeddings - mean) / std

    embeddings = F.normalize(embeddings, p=2, dim=1)

    similarities = (embeddings @ embeddings.T).clamp(-1, 1)
    distances = (1.0 - similarities)

    # check that distances make sense
    eps = 1e-5
    diag = torch.diag(distances)
    assert diag.allclose(torch.zeros_like(diag), rtol=eps, atol=eps)

    return distances

def distances_pearson(A: Tensor, B: Tensor) -> float:
    eps = 1e-8
    assert A.shape == B.shape

    V = A.size(0)
    iu = torch.triu_indices(V, V, offset=1) # upper triangle no diag

    a = A[iu[0], iu[1]].view(-1)
    b = B[iu[0], iu[1]].view(-1)

    a = a - a.mean()
    b = b - b.mean()
    numerator = (a * b).sum()
    denominator = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()).clamp_min(eps)

    pearson = numerator / denominator
    return pearson.item()

def load_model(type: str, id: int) -> BertModel:
    assert type in ['llm', 'glm']
    name = f'{type}_{id}'
    model_dir = get_models_path() / name
    model = BertForMaskedLM.from_pretrained(str(model_dir.resolve()), local_files_only=True).eval().bert
    return model

def load_embeddings(type: str, id: int) -> Tensor:
    model = load_model(type, id)
    embeddings = model.embeddings.word_embeddings.weight.detach()
    return embeddings

@torch.no_grad()
def main():

    print('GLMs:')

    distance_matrices = []

    for i in range(2):
        embeddings = load_embeddings('glm', i)
        distances = cosine_distances(embeddings)
        distance_matrices.append(distances)

    r = distances_pearson(*distance_matrices)
    print(r)

    cka = linear_cka(*distance_matrices)
    print(cka)

    print('LLMs:')

    distance_matrices = []

    for i in range(2):
        embeddings = load_embeddings('llm', i)
        distances = cosine_distances(embeddings)
        distance_matrices.append(distances)

    r = distances_pearson(*distance_matrices)
    print(r)

    cka = linear_cka(*distance_matrices)
    print(cka)

if __name__ == '__main__': main()
