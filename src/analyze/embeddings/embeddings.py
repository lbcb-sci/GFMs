import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.analyze.data import mlm_preprocess, get_dataset_wiki, get_dna_dataset, DeviceWrapper
from src.analyze import metrics as m
from src.utils import DATA_TOKENIZER_PAIRS, create_results_dict, run_key

HIGHER_IS_BETTER = {
    'effective_rank': True,
    'uniformity': False,
    'anisotropy': False,
    'avgcosim': False,
    'pairwise_dist_std': True,
    'neighbor_alignment': True,
    'alignment_l2': False,
    'linear_cka': True,
}

def print_metric_rankings(results: dict):
    # flat = {}
    # for key in results:
    #     flat[key] = results[key]

    metrics = list(next(iter(results.values())).keys())

    for metric in metrics:
        print(f"\n=== {metric} ===")

        entries = []
        for name, vals in results.items():
            mean, std = vals[metric]
            entries.append((name, mean, std))

        reverse = HIGHER_IS_BETTER.get(metric, True)

        ranked = sorted(entries, key=lambda x: x[1], reverse=reverse)

        for i, (name, mean, std) in enumerate(ranked):
            print(f"{i+1:2d}. {name:20s} : {mean:.4f} ± {std:.4f}")

# create a folder /src/analyze/embeddings and put this file inside 
# then call it from /src/analyze/__main__.py

@torch.inference_mode()
def embeddings(all_models: dict, tokenizers: dict, args) -> dict:
    logger, n_samples, batch_size = args.logger, args.samples, args.batch_size

    results = create_results_dict()

    logger.info(f' computing metrics over embeddings...')

    pr = create_results_dict()

    for data, tok, type in DATA_TOKENIZER_PAIRS:
        key = run_key(data, tok, type)
        logger.info(f' extracting last-layer contextual embeddings for {key}...')

        models = all_models[key]
        tokenizer = tokenizers[key][0]

        if data == 'text':
            dataset = get_dataset_wiki(n_samples, preprocessed=True)
        else:
            dataset = get_dna_dataset(type, n_samples)

        remove = ['text']
        for col in ['url', 'id', 'title']:
            if col in dataset.column_names:
                remove.append(col)

        logger.info(' masking tokens in dataset...')
        preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.15)
        encoded = dataset.map(preprocess, batched=True, remove_columns=remove, load_from_cache_file=False)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        [model.to(args.device) for model in models]


        dataloader = DataLoader(
            DeviceWrapper(encoded, device=args.device),
            batch_size=batch_size,
            shuffle=False,
        )

        embeddings = get_embeddings(models, dataloader)
        result = compare_embeddings(embeddings)
        results[key] = result
        print(key)
        print(result)
        print()

        [model.cpu() for model in models]

    print_metric_rankings(results)

    return results

def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    sum_hidden = (hidden_states * mask).sum(dim=1)
    n_tokens = mask.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / n_tokens

def get_embeddings(models, dataloader):
    result = []

    for model in models:
        model_embeddings = []

        for batch in tqdm(dataloader):
            output = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            emb = output.hidden_states[-1]
            mask = batch['attention_mask']
            pooled = mean_pool(emb, mask)
            model_embeddings.extend(pooled.cpu())

        result.append(torch.stack(model_embeddings))

    result = torch.stack(result)
    return result

def uniformity(X):
    X_norm = F.normalize(X, dim=-1)
    dists = torch.pdist(X_norm, p=2).pow(2)
    return dists.mul(-2).exp().mean().log().item()

def participation_ratio(X):
    X_centered = X - X.mean(dim=0)
    _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    S2 = S**2
    return (S2.sum()**2 / (S2**2).sum()).item()

def avg_cosine_sim(X):
    X_norm = F.normalize(X, dim=-1)
    sim = X_norm @ X_norm.T
    idx = torch.triu_indices(len(X), len(X), offset=1)
    return sim[idx[0], idx[1]].mean().item()

def neighbor_alignment(X, Y, k=10):
    X_norm = F.normalize(X, dim=-1)
    Y_norm = F.normalize(Y, dim=-1)

    def get_knn(Z):
        sim = Z @ Z.T
        sim.fill_diagonal_(float('-inf'))
        return sim.topk(k, dim=-1).indices

    nn_x = get_knn(X_norm)
    nn_y = get_knn(Y_norm)

    shared = sum(
        len(set(nn_x[i].tolist()) & set(nn_y[i].tolist()))
        for i in range(len(X))
    )
    return shared / (len(X) * k)

def alignment(X, Y):
    X_norm = F.normalize(X, dim=-1)
    Y_norm = F.normalize(Y, dim=-1)
    return (X_norm - Y_norm).norm(dim=-1).pow(2).mean().item()

def effective_rank(X):
    X_centered = X - X.mean(dim=0)
    _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    p = S / S.sum()
    entropy = -(p * torch.log(p + 1e-9)).sum()
    return torch.exp(entropy).item()

def anisotropy(X):
    X_norm = F.normalize(X, dim=-1)
    mean_vec = X_norm.mean(dim=0)
    return mean_vec.norm().item()

def pairwise_dist_stats(X):
    dists = torch.pdist(X, p=2)
    return dists.mean().item(), dists.std().item()

def linear_cka(X, Y):
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    num   = YtX.norm(p='fro').pow(2)
    denom = XtX.norm(p='fro') * YtY.norm(p='fro')

    return (num / denom).item()

def compare_embeddings(embeddings):
    import numpy as np

    unif_vals, cosim_vals, aniso_vals, effrank_vals, dist_std_vals = [], [], [], [], []
    nb_align_vals, align_vals, cka_vals = [], [], []

    for i, a in enumerate(embeddings):
        unif_vals.append(uniformity(a))
        cosim_vals.append(avg_cosine_sim(a))
        aniso_vals.append(anisotropy(a))
        effrank_vals.append(effective_rank(a))
        _, ds = pairwise_dist_stats(a)
        dist_std_vals.append(ds)

        for b in embeddings[i+1:]:
            nb_align_vals.append(neighbor_alignment(a, b))
            align_vals.append(alignment(a, b))
            cka_vals.append(linear_cka(a, b))

    def ms(lst): return (float(np.mean(lst)), float(np.std(lst, ddof=1)))

    return {
        'effective_rank':     ms(effrank_vals),
        'uniformity':         ms(unif_vals),
        'anisotropy':         ms(aniso_vals),
        'avgcosim':           ms(cosim_vals),
        'pairwise_dist_std':  ms(dist_std_vals),
        'neighbor_alignment': ms(nb_align_vals),
        'alignment_l2':       ms(align_vals),
        'linear_cka':         ms(cka_vals),
    }

@torch.inference_mode()
def layerwise_PR(models, dataloader, device):
    n_layers = models[0].config.num_hidden_layers + 1
    results = []

    for layer_idx in range(n_layers):
        layer_prs = []

        for model in tqdm(models):
            model.to(device)
            model_embs = []

            for batch in dataloader:
                output = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    output_hidden_states=True
                )
                pooled = mean_pool(
                    output.hidden_states[layer_idx],
                    batch['attention_mask']
                )
                model_embs.append(pooled.cpu())

            embs = torch.cat(model_embs)
            layer_prs.append(participation_ratio(embs))
            model.cpu()

        prs = torch.tensor(layer_prs)
        results.append({
            "layer":   layer_idx,
            "mean_pr": prs.mean().item(),
            "std_pr":  prs.std().item(),
            "all_prs": prs,
        })

    return results

import matplotlib.pyplot as plt
import seaborn as sns

def plot_layerwise_pr(results_per_group: dict):

    sns.set_theme(
        context='paper',
        style='white',
        palette='pastel',
    )

    colors = ["steelblue", "tomato", "seagreen"]
    fig, ax = plt.subplots(figsize=(10, 4))

    for (data, tok, type), color in zip(DATA_TOKENIZER_PAIRS, colors):

        key = run_key(data, tok, type)
        layers = torch.tensor([r["layer"]   for r in results_per_group[key]])
        means  = torch.tensor([r["mean_pr"] for r in results_per_group[key]])
        stds   = torch.tensor([r["std_pr"]  for r in results_per_group[key]])

        ax.plot(layers.tolist(), means.tolist(), label=f'{data}-{tok}', color=color)
        ax.fill_between(
            layers.tolist(),
            (means - stds).tolist(),
            (means + stds).tolist(),
            alpha=0.2, color=color
        )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation Ratio")
    ax.legend()
    plt.tight_layout()
    plt.savefig('pr.pdf')
    #return fig
