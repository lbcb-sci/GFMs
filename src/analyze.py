import os
import torch 
import numpy as np

from src.utils.utils import get_models_path, load_many_embeddings
from src import metrics

@torch.no_grad()
def main():

    mean_std = lambda data: (np.mean(data), np.std(data))

    llms = []; glms_bpe = []; glms_overlapping = []

    models_path = get_models_path()
    ls = os.listdir(models_path)
    for d in ls: 
        if   'llm' in d and '512' in d: llms.append(models_path / d)
        elif 'glm' in d and 'bpe' in d: glms_bpe.append(models_path / d)
        elif 'glm' in d and 'overlapping' in d: glms_overlapping.append(models_path / d)
        else: pass

    glms_bpe_embeddings = load_many_embeddings(glms_bpe)
    glms_ol_embeddings = load_many_embeddings(glms_overlapping)
    llms_embeddings = load_many_embeddings(llms)

    print(len(glms_bpe_embeddings), len(glms_ol_embeddings), len(llms_embeddings))

    glms_bpe_sims = [metrics.cosine_similarities(emb) for emb in glms_bpe_embeddings]
    glms_ol_sims = [metrics.cosine_similarities(emb) for emb in glms_ol_embeddings]
    llms_sims = [metrics.cosine_similarities(emb) for emb in llms_embeddings]

    ### TOP-3 OVERLAP

    glm_top3_overlap_bpe = metrics.topk_neighbor_overlap(glms_bpe_sims, k=3)
    glm_top3_overlap_ol = metrics.topk_neighbor_overlap(glms_ol_sims, k=3)
    llm_top3_overlap = metrics.topk_neighbor_overlap(llms_sims, k=3)

    glm_top3_overlap_bpe_mean, glm_top3_overlap_bpe_std = mean_std(glm_top3_overlap_bpe)
    glm_top3_overlap_ol_mean, glm_top3_overlap_ol_std = mean_std(glm_top3_overlap_ol)
    llm_top3_overlap_mean, llm_top3_overlap_std = mean_std(llm_top3_overlap)

    print(f'GLM BPE Top-3 Overlap: {glm_top3_overlap_bpe_mean:.2f} ({glm_top3_overlap_bpe_std:.3f})')
    print(f'GLM OL Top-3 Overlap: {glm_top3_overlap_ol_mean:.2f} ({glm_top3_overlap_ol_std:.3f})')
    print(f'LLM Top-3 Overlap: {llm_top3_overlap_mean:.2f} ({llm_top3_overlap_std:.3f})')
    print()

    ### TOP-10 OVERLAP

    glm_top10_overlap_bpe = metrics.topk_neighbor_overlap(glms_bpe_sims, k=10)
    glm_top10_overlap_ol = metrics.topk_neighbor_overlap(glms_ol_sims, k=10)
    llm_top10_overlap = metrics.topk_neighbor_overlap(llms_sims, k=10)

    glm_top10_overlap_bpe_mean, glm_top10_overlap_bpe_std = mean_std(glm_top10_overlap_bpe)
    glm_top10_overlap_ol_mean, glm_top10_overlap_ol_std = mean_std(glm_top10_overlap_ol)
    llm_top10_overlap_mean, llm_top10_overlap_std = mean_std(llm_top10_overlap)

    print(f'GLM BPE Top-10 Overlap: {glm_top10_overlap_bpe_mean:.2f} ({glm_top10_overlap_bpe_std:.3f})')
    print(f'GLM OL Top-10 Overlap: {glm_top10_overlap_ol_mean:.2f} ({glm_top10_overlap_ol_std:.3f})')
    print(f'LLM Top-10 Overlap: {llm_top10_overlap_mean:.2f} ({llm_top10_overlap_std:.3f})')
    print()

    ### TOP-100 OVERLAP

    glm_top100_overlap_bpe = metrics.topk_neighbor_overlap(glms_bpe_sims, k=100)
    glm_top100_overlap_ol = metrics.topk_neighbor_overlap(glms_ol_sims, k=100)
    llm_top100_overlap = metrics.topk_neighbor_overlap(llms_sims, k=100)

    glm_top100_overlap_bpe_mean, glm_top100_overlap_bpe_std = mean_std(glm_top100_overlap_bpe)
    glm_top100_overlap_ol_mean, glm_top100_overlap_ol_std = mean_std(glm_top100_overlap_ol)
    llm_top100_overlap_mean, llm_top100_overlap_std = mean_std(llm_top100_overlap)

    print(f'GLM BPE Top-100 Overlap: {glm_top100_overlap_bpe_mean:.2f} ({glm_top100_overlap_bpe_std:.3f})')
    print(f'GLM OL Top-100 Overlap: {glm_top100_overlap_ol_mean:.2f} ({glm_top100_overlap_ol_std:.3f})')
    print(f'LLM Top-100 Overlap: {llm_top100_overlap_mean:.2f} ({llm_top100_overlap_std:.3f})')
    print()

    ### LOCAL SPEARMAN

    glm_local_spearman_bpe = metrics.local_spearman(glms_bpe_sims)
    glm_local_spearman_ol = metrics.local_spearman(glms_ol_sims)
    llm_local_spearman = metrics.local_spearman(llms_sims)

    glm_local_spearman_bpe_mean, glm_local_spearman_bpe_std = mean_std(glm_local_spearman_bpe)
    glm_local_spearman_ol_mean, glm_local_spearman_ol_std = mean_std(glm_local_spearman_ol)
    llm_local_spearman_mean, llm_local_spearman_std = mean_std(llm_local_spearman)

    print(f'GLM BPE Local Spearman: {glm_local_spearman_bpe_mean:.2f} ({glm_local_spearman_bpe_std:.3f})')
    print(f'GLM OL Local Spearman: {glm_local_spearman_ol_mean:.2f} ({glm_local_spearman_ol_std:.3f})')
    print(f'LLM Local Spearman: {llm_local_spearman_mean:.2f} ({llm_local_spearman_std:.3f})')
    print()

    ### STDEV

    relative_std_bpe = metrics.relative_diff_std(glms_bpe_sims, llms_sims)
    relative_std_ol = metrics.relative_diff_std(glms_ol_sims, llms_sims)

    llm_per_token_std = metrics.per_token_std(llms_sims).mean()
    glm_per_token_std = metrics.per_token_std(glms_bpe_sims).mean()

    print(f'GLM Mean std per token across models: {glm_per_token_std:.4f}')
    print(f'LLM Mean std per token across models: {llm_per_token_std:.4f}')
    print()
    print(f'BPE Relative std difference: +{relative_std_bpe*100:.0f}%')
    print(f'OL Relative std difference: +{relative_std_ol*100:.0f}%')
    print()

    ### CKA

    llm_ckas = metrics.cka(llms_embeddings)
    glm_ckas_bpe = metrics.cka(glms_bpe_embeddings)
    glm_ckas_ol = metrics.cka(glms_ol_embeddings)

    llm_cka_mean, llm_cka_std = mean_std(llm_ckas)
    glm_cka_bpe_mean, glm_cka_bpe_std = mean_std(glm_ckas_bpe)
    glm_cka_ol_mean, glm_cka_ol_std = mean_std(glm_ckas_ol)

    print(f'GLM BPE CKA: {glm_cka_bpe_mean:.2f} ({glm_cka_bpe_std:.3f})')
    print(f'GLM OL CKA: {glm_cka_ol_mean:.2f} ({glm_cka_ol_std:.3f})')
    print(f'LLM CKA: {llm_cka_mean:.2f} ({llm_cka_std:.3f})')

if __name__ == '__main__': main()
