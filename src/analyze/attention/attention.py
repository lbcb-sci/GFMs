import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

from src.analyze.data import mlm_preprocess, get_opengenome, get_wikipedia, DeviceWrapper
from src.utils import N, DATA_TOKENIZER_PAIRS, create_results_dict

def attention(all_models: dict, tokenizers: dict, args) -> dict:
    logger, n_samples, batch_size = args.logger, args.samples, args.batch_size

    logger.info(f' computing metrics over attention scores...')

    results = create_results_dict()

    for data, tok in DATA_TOKENIZER_PAIRS:

        logger.info(f' computing for ({data}, {tok})...')

        models = all_models[data][tok]#[:2]
        tokenizer = tokenizers[data][tok][0]

        assert all([model.name_or_path[:-1] == tokenizer.name_or_path[:-1] for model in models])

        is_text = 'text' in tokenizer.name_or_path

        logger.info(f' collecting dataset {"text" if is_text else "dna"}...')
        dataset = get_wikipedia(n_samples) if is_text else get_opengenome(n_samples)

        remove = ['text', 'url', 'id', 'title'] if is_text else ['text']

        logger.info( 'masking tokens in dataset...')
        preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.10)
        encoded = dataset.map(preprocess, batched=True, remove_columns=remove, load_from_cache_file=False)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dataloader = DataLoader(
            DeviceWrapper(encoded, device=args.device),
            batch_size=batch_size,
            shuffle=False,
        )

        mifull, mismall, entropies = get_mimatrix(models, dataloader, logger)

        results[data][tok][f'mimatrix'] = mismall
        results[data][tok][f'mimatrix_full'] = mifull
        results[data][tok][f'entropies'] = entropies

        [model.cpu() for model in models]

    return results

@torch.autograd.inference_mode()
def get_mimatrix(models, dataloader, logger):
    num_layers = 12
    num_models = len(models)

    all_masked_scores = [[[] for _ in range(num_layers)] for _ in range(len(models))]

    entropy_sum = torch.zeros(num_layers)
    entropy_count = torch.zeros(num_layers)

    for batch in tqdm(dataloader, desc="Extracting Scores"):
        mask = (batch['labels'] != -100) # [B, S]

        num_masked = mask.sum().item()

        row_mask = mask.unsqueeze(1).unsqueeze(-1) 

        for m_idx, model in enumerate(models):
            outputs = model(**batch).attentions 

            for l_idx, layer_attn in enumerate(outputs):
                masked_attn = layer_attn.transpose(1, 2)[mask]
                token_head_entropy = -torch.sum(masked_attn * torch.log(masked_attn + 1e-10), dim=-1)
                token_entropy = token_head_entropy.mean(dim=-1)
                entropy_sum[l_idx] += token_entropy.sum().cpu()
                entropy_count[l_idx] += num_masked

                valid_scores = layer_attn[row_mask.expand_as(layer_attn)]
                all_masked_scores[m_idx][l_idx].append(valid_scores.cpu())

    entropies = (entropy_sum / entropy_count)
    entropies = {l: e for l, e in enumerate(entropies)}

    final_vectors = []
    for m_idx in range(len(models)):
        layers = [torch.cat(all_masked_scores[m_idx][l]) for l in range(num_layers)]
        final_vectors.append(layers)

    logger.info(' bucketizing all vectors...')
    final_vectors = bucketize_all(final_vectors)
    logger.info(' done.')
    
    mi_matrix_full = np.zeros((num_layers*num_models, num_layers*num_models))
    mi_matrix_small = np.zeros((num_layers, num_layers))

    for m1 in tqdm(range(num_models)):
        for m2 in range(num_models):
            for l1 in range(num_layers):
                for l2 in range(num_layers):

                    if (m1 * num_layers + l1) > (m2 * num_layers + l2):
                        continue  # only compute upper triangle, mirror below

                    mi_score = nmi(final_vectors[m1][l1], final_vectors[m2][l2]).item()

                    mi_matrix_full[l1*num_models+m1, l2*num_models+m2] = mi_score
                    mi_matrix_full[l2*num_models+m2, l1*num_models+m1] = mi_score

                    if m1 == m2: continue
                    mi_matrix_small[l1, l2] = max(mi_score, mi_matrix_small[l1, l2])
                    mi_matrix_small[l2, l1] = max(mi_score, mi_matrix_small[l2, l1])

    #maxval = mi_matrix_full.max()
    #for i in range(mi_matrix_full.shape[0]):
        #mi_matrix_full[i, i] = maxval
                
    return mi_matrix_full, mi_matrix_small, entropies

def bucketize(vec: Tensor) -> Tensor:
    N = vec.numel()
    bins = int(torch.sqrt(torch.tensor(N / 10)).item())
    vec_bins = torch.bucketize(vec, torch.linspace(vec.min(), vec.max(), bins+1, device=vec.device))
    return vec_bins

def bucketize_all(final_vectors):
    for i, v in enumerate(final_vectors):
        for j, vv in enumerate(v):
            final_vectors[i][j] = bucketize(vv)
    return final_vectors

def mutualinf(p_bins, q_bins):
    num_bins = int(max(p_bins.max(), q_bins.max()).item()) + 1

    joint_indices = p_bins * num_bins + q_bins

    joint_probs = torch.bincount(joint_indices, minlength=num_bins**2).float() / p_bins.numel()
    joint_probs = joint_probs.reshape(num_bins, num_bins)

    p_probs = joint_probs.sum(dim=1)
    q_probs = joint_probs.sum(dim=0)

    outer = p_probs.unsqueeze(1) * q_probs.unsqueeze(0)  # (num_bins, num_bins)
    nz = joint_probs > 0
    mi = torch.sum(joint_probs[nz] * torch.log(joint_probs[nz] / outer[nz]))
    return mi

def sym_mi(p_bins, q_bins):
    return (mutualinf(p_bins, q_bins) + mutualinf(q_bins, p_bins)) / 2

def nmi(p_bins, q_bins):
    mi = sym_mi(p_bins, q_bins)

    # marginal entropies
    def marginal_entropy(bins):
        num_bins = int(bins.max().item()) + 1
        probs = torch.bincount(bins, minlength=num_bins).float() / bins.numel()
        nz = probs > 0
        return -torch.sum(probs[nz] * torch.log(probs[nz])).item()

    hx = marginal_entropy(p_bins)
    hy = marginal_entropy(q_bins)

    denom = torch.sqrt(hx * hy)
    if denom < 1e-10: return 0.0

    return mi / denom
