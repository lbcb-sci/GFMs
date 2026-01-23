import os
import torch 
import numpy as np
from torch import Tensor
from transformers import BertForMaskedLM, BertModel

from src.common import get_models_path, print_parameters
from src.instability.metrics import *

def load_model(path) -> BertModel:
    return BertForMaskedLM.from_pretrained(str(path.resolve()), local_files_only=True).eval().bert

def load_embeddings(path) -> Tensor:
    model = load_model(path)
    print_parameters(model)
    embeddings = model.embeddings.word_embeddings.weight.detach()
    return embeddings

def load_many_embeddings(paths: list) -> list[Tensor]:
    result = []
    for path in paths:
        emb = load_embeddings(path)
        result.append(emb)
    return result

def mean_std(data) -> tuple[float, float]:
    return np.mean(data), np.std(data)

def linear_cka(emb1: Tensor, emb2: Tensor) -> float:
    Xc = emb1 - emb1.mean(dim=0, keepdim=True)
    Yc = emb2 - emb2.mean(dim=0, keepdim=True)
    K, L = Xc @ Xc.T, Yc @ Yc.T
    return (K * L).sum() / (torch.norm(K, p='fro') * torch.norm(L, p='fro')).item()

def cka(embeddings) -> float:
    results = []
    for i, a in enumerate(embeddings):
        for b in embeddings[i+1:]:
            results.append(linear_cka(a, b))
    return results

@torch.no_grad()
def main():

    models_path = get_models_path()
    ls = os.listdir(models_path)

    llms = []; glms_bpe = []; glms_overlapping = []

    for d in ls: 
        if   'llm' in d: llms.append(models_path / d)
        elif 'glm' in d and 'bpe' in d: glms_bpe.append(models_path / d)
        elif 'glm' in d and 'overlapping' in d: glms_overlapping.append(models_path / d)
        else: pass

    glms_bpe_embeddings = load_many_embeddings(glms_bpe)
    glms_ol_embeddings = load_many_embeddings(glms_overlapping)
    llms_embeddings = load_many_embeddings(llms)

    print(len(glms_bpe_embeddings), len(glms_ol_embeddings), len(llms_embeddings))

    glms_bpe_sims = [cosine_similarities(emb) for emb in glms_bpe_embeddings]
    glms_ol_sims = [cosine_similarities(emb) for emb in glms_ol_embeddings]
    llms_sims = [cosine_similarities(emb) for emb in llms_embeddings]

    glm_top3_overlap_bpe = topk_neighbor_overlap(glms_bpe_sims, k=3)
    glm_top3_overlap_ol = topk_neighbor_overlap(glms_ol_sims, k=3)
    llm_top3_overlap = topk_neighbor_overlap(llms_sims, k=3)

    glm_top3_overlap_bpe_mean, glm_top3_overlap_bpe_std = mean_std(glm_top3_overlap_bpe)
    glm_top3_overlap_ol_mean, glm_top3_overlap_ol_std = mean_std(glm_top3_overlap_ol)
    llm_top3_overlap_mean, llm_top3_overlap_std = mean_std(llm_top3_overlap)

    glm_top10_overlap_bpe = topk_neighbor_overlap(glms_bpe_sims, k=10)
    glm_top10_overlap_ol = topk_neighbor_overlap(glms_ol_sims, k=10)
    llm_top10_overlap = topk_neighbor_overlap(llms_sims, k=10)

    glm_top10_overlap_bpe_mean, glm_top10_overlap_bpe_std = mean_std(glm_top10_overlap_bpe)
    glm_top10_overlap_ol_mean, glm_top10_overlap_ol_std = mean_std(glm_top10_overlap_ol)
    llm_top10_overlap_mean, llm_top10_overlap_std = mean_std(llm_top10_overlap)

    glm_top100_overlap_bpe = topk_neighbor_overlap(glms_bpe_sims, k=100)
    glm_top100_overlap_ol = topk_neighbor_overlap(glms_ol_sims, k=100)
    llm_top100_overlap = topk_neighbor_overlap(llms_sims, k=100)

    glm_top100_overlap_bpe_mean, glm_top100_overlap_bpe_std = mean_std(glm_top100_overlap_bpe)
    glm_top100_overlap_ol_mean, glm_top100_overlap_ol_std = mean_std(glm_top100_overlap_ol)
    llm_top100_overlap_mean, llm_top100_overlap_std = mean_std(llm_top100_overlap)

    glm_local_spearman_bpe = local_spearman_sim(glms_bpe_sims)
    glm_local_spearman_ol = local_spearman_sim(glms_ol_sims)
    llm_local_spearman = local_spearman_sim(llms_sims)

    glm_local_spearman_bpe_mean, glm_local_spearman_bpe_std = mean_std(glm_local_spearman_bpe)
    glm_local_spearman_ol_mean, glm_local_spearman_ol_std = mean_std(glm_local_spearman_ol)
    llm_local_spearman_mean, llm_local_spearman_std = mean_std(llm_local_spearman)

    relative_std_bpe = relative_diff_std(glms_bpe_sims, llms_sims)
    relative_std_ol = relative_diff_std(glms_ol_sims, llms_sims)

    print(f'GLM BPE Top-3 Overlap: {glm_top3_overlap_bpe_mean:.2f} ({glm_top3_overlap_bpe_std:.3f})')
    print(f'GLM OL Top-3 Overlap: {glm_top3_overlap_ol_mean:.2f} ({glm_top3_overlap_ol_std:.3f})')
    print(f'LLM Top-3 Overlap: {llm_top3_overlap_mean:.2f} ({llm_top3_overlap_std:.3f})')

    print()

    print(f'GLM BPE Top-10 Overlap: {glm_top10_overlap_bpe_mean:.2f} ({glm_top10_overlap_bpe_std:.3f})')
    print(f'GLM OL Top-10 Overlap: {glm_top10_overlap_ol_mean:.2f} ({glm_top10_overlap_ol_std:.3f})')
    print(f'LLM Top-10 Overlap: {llm_top10_overlap_mean:.2f} ({llm_top10_overlap_std:.3f})')

    print()

    print(f'GLM BPE Top-100 Overlap: {glm_top100_overlap_bpe_mean:.2f} ({glm_top100_overlap_bpe_std:.3f})')
    print(f'GLM OL Top-100 Overlap: {glm_top100_overlap_ol_mean:.2f} ({glm_top100_overlap_ol_std:.3f})')
    print(f'LLM Top-100 Overlap: {llm_top100_overlap_mean:.2f} ({llm_top100_overlap_std:.3f})')

    print()

    print(f'GLM BPE Local Spearman: {glm_local_spearman_bpe_mean:.2f} ({glm_local_spearman_bpe_std:.3f})')
    print(f'GLM OL Local Spearman: {glm_local_spearman_ol_mean:.2f} ({glm_local_spearman_ol_std:.3f})')
    print(f'LLM Local Spearman: {llm_local_spearman_mean:.2f} ({llm_local_spearman_std:.3f})')

    print()

    llm_per_token_std = per_token_std(llms_sims).mean()
    glm_per_token_std = per_token_std(glms_bpe_sims).mean()

    print(f'GLM Mean std per token across models: {glm_per_token_std:.4f}')
    print(f'LLM Mean std per token across models: {llm_per_token_std:.4f}')

    print()

    print(f'BPE Relative std difference: +{relative_std_bpe*100:.0f}%')
    print(f'OL Relative std difference: +{relative_std_ol*100:.0f}%')

    print()

    llm_ckas = cka(llms_embeddings)
    glm_ckas_bpe = cka(glms_bpe_embeddings)
    glm_ckas_ol = cka(glms_ol_embeddings)

    llm_cka_mean, llm_cka_std = mean_std(llm_ckas)
    glm_cka_bpe_mean, glm_cka_bpe_std = mean_std(glm_ckas_bpe)
    glm_cka_ol_mean, glm_cka_ol_std = mean_std(glm_ckas_ol)

    print(f'GLM BPE CKA: {glm_cka_bpe_mean:.2f} ({glm_cka_bpe_std:.3f})')
    print(f'GLM OL CKA: {glm_cka_ol_mean:.2f} ({glm_cka_ol_std:.3f})')
    print(f'LLM CKA: {llm_cka_mean:.2f} ({llm_cka_std:.3f})')

if __name__ == '__main__': main()
