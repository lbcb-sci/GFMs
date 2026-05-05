import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.analyze.data import mlm_preprocess, get_dataset_wiki, get_dna_dataset, DeviceWrapper
from src.utils import DATA_TOKENIZER_PAIRS, create_results_dict, run_key

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

        #l = layerwise_PR(models, dataloader, args.device)
        #pr[key] = l
        #print(l)

        embeddings = get_embeddings(models, dataloader)
        result = compare_embeddings(embeddings)
        results[key] = result
        print(key)
        print(result)
        print()

        [model.cpu() for model in models]

    #plot_layerwise_pr(pr)

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

def mean_cosine_similarity(X, Y):
    X_norm = F.normalize(X, dim=-1)
    Y_norm = F.normalize(Y, dim=-1)
    return (X_norm * Y_norm).sum(dim=-1).mean().item()

def uniformity(X):
    X_norm = F.normalize(X, dim=-1)
    dists = torch.pdist(X_norm, p=2).pow(2)
    return dists.mul(-2).exp().mean().log().item()

def isotropy(X):
    X_centered = X - X.mean(dim=0)
    cov = X_centered.T @ X_centered / len(X)
    eigvals = torch.linalg.eigvalsh(cov)      # sorted ascending
    return (eigvals[-1] / eigvals[0].clamp(min=1e-9)).log().item()

def participation_ratio(X):
    # Softer version — doesn't require threshold choice
    X_centered = X - X.mean(dim=0)
    _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
    S2 = S**2
    return (S2.sum()**2 / (S2**2).sum()).item()

def avg_cosine_sim(X):
    # Measures how much embeddings agree in direction on average
    # High (~1.0) = anisotropic, Low (~0.0) = well spread
    X_norm = F.normalize(X, dim=-1)
    sim = X_norm @ X_norm.T                    # (n, n)
    idx = torch.triu_indices(len(X), len(X), offset=1)
    return sim[idx[0], idx[1]].mean().item()

def neighbor_alignment(X, Y, k=5):
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

def compare_embeddings(embeddings):
    nb_alignment = 0.0
    unif = 0.0
    partratio = 0.0
    avgcosim = 0.0

    count = 0

    for i, a in enumerate(embeddings):
        unif += uniformity(a)
        partratio += participation_ratio(a)
        avgcosim += avg_cosine_sim(a)
        for b in embeddings[i+1:]:
            nb_alignment += neighbor_alignment(a, b)
            count += 1
    
    return {
        'participation_ratio': partratio / embeddings.shape[0],
        'uniformity': unif / embeddings.shape[0],
        'avgcosim': avgcosim / embeddings.shape[0],
        'neighbor_alignment': nb_alignment / count,
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

    colors = ["steelblue", "tomato", "seagreen", "purple", "orange", "brown", "pink"]
    fig, ax = plt.subplots(figsize=(10, 4))

    for (data, tok, type), color in zip(DATA_TOKENIZER_PAIRS, colors):
        key = run_key(data, tok, type)

        layers = torch.tensor([r["layer"]   for r in results_per_group[key]])
        means  = torch.tensor([r["mean_pr"] for r in results_per_group[key]])
        stds   = torch.tensor([r["std_pr"]  for r in results_per_group[key]])

        ax.plot(layers.tolist(), means.tolist(), label=key, color=color)
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