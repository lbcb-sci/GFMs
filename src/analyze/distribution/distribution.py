import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

from src.analyze.data import get_dataset_wiki, get_dna_dataset, mlm_preprocess, DeviceWrapper
from src.analyze.metrics import kl_divergence, jensen_shannon_distance
from src.utils import DATA_TOKENIZER_PAIRS, create_results_dict, run_key

def distributions(all_models: dict, tokenizers: dict, args) -> dict:
    logger, n_samples, batch_size = args.logger, args.samples, args.batch_size

    TOP_P_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    logger.info(f' computing metrics over output distributions...')

    results = create_results_dict()

    for data, tok, type in DATA_TOKENIZER_PAIRS:
        key = run_key(data, tok, type)
        models = all_models[key]
        tokenizer = tokenizers[key][0]

        assert all([model.name_or_path[:-1] == tokenizer.name_or_path[:-1] for model in models])

        logger.info(f' collecting dataset for {key}...')
        if data == 'text':
            dataset = get_dataset_wiki(n_samples, preprocessed=True)
        else:
            dataset = get_dna_dataset(type, n_samples)

        remove = ['text']
        if 'url' in dataset.column_names: remove.append('url')
        if 'id' in dataset.column_names: remove.append('id')
        if 'title' in dataset.column_names: remove.append('title')

        logger.info('masking tokens in dataset...')
        preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.10)
        encoded = dataset.map(preprocess, batched=True, remove_columns=remove, load_from_cache_file=False)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        [model.to(args.device) for model in models]

        dataloader = DataLoader(
            DeviceWrapper(encoded, device=args.device),
            batch_size=batch_size,
            shuffle=False,
        )

        distributions, kl_values = get_distributions(models, dataloader, logger)
        assert (distributions.sum(dim=-1) - 1 < 1e-5).all()

        mean_dist = compute_mean_distribution(distributions)
        results[key]['mean_dist'] = mean_dist.cpu()
        results[key]['kl'] = kl_values

        # jensen_shannon = compute_jensen_shannon(distributions, TOP_P_VALUES, logger)
        # results[key]['js'] = jensen_shannon

        [model.cpu() for model in models]
        print()

    return results

def compute_mean_distribution(distributions: dict[int, Tensor]) -> Tensor:
    sorted_dists, _ = torch.sort(distributions, dim=-1, descending=True)
    mean_sorted_dist = sorted_dists.mean(axis=(0, 1))
    return mean_sorted_dist

def compute_jensen_shannon(
    distributions: dict[int, Tensor],
    top_p_values: list[float],
    logger,
) -> dict[float, float]:
    desc = f'computing jensen-shannon for {len(top_p_values)} p-values'

    total_values = 0

    result = {}
    for p in tqdm(top_p_values, desc=desc):

        total = jsd_p = 0
        dists_top_p = top_p_reweight(distributions, p)

        for i, a in enumerate(dists_top_p):
            for b in dists_top_p[i+1:]:

                jsd = jensen_shannon_distance(a, b)

                nan = torch.isnan(jsd)
                nan_mean = jsd[~nan].mean()
                jsd[nan] = nan_mean

                total_values += jsd.numel()
                jsd_p += jsd.mean()

                total += 1

        result[p] = float(jsd_p / total)

    logger.info(f' jensen-shannon total values = {total_values:,}')
    return result

@torch.autograd.inference_mode()
def get_distributions(
    models: tuple[BertForMaskedLM],
    dataloader: DataLoader,
    logger,
) -> tuple[Tensor, list[float]]:
    '''Extract distributions over masked tokens. Also computes KL div, perplexity and accuracy.'''

    result = []
    kl_values = []

    for i, model in enumerate(models):

        total_nll = 0.0
        total_kl = 0.0
        total_correct_predictions = 0
        total_masked_tokens = 0
        dists_model = []

        for batch in tqdm(dataloader):
            labels = batch['labels']

            mask = (labels != -100)
            labels = labels[mask]

            num_masked_tokens = labels.numel()
            total_masked_tokens += num_masked_tokens

            logits = model(**batch).logits[mask]
            assert logits.shape[0] == num_masked_tokens

            total_nll += F.cross_entropy(logits, labels, reduction='sum')

            distributions = F.softmax(logits, dim=1)
            assert (distributions.sum(dim=1) - 1 < 1e-5).all()

            uniform = torch.ones_like(distributions) / distributions.shape[-1]
            total_kl += kl_divergence(uniform, distributions).sum()

            dists_model.extend(distributions.cpu())
            total_correct_predictions += (distributions.argmax(-1) == labels).float().sum()

        dists_model = torch.stack(dists_model)
        assert dists_model.shape[0] == total_masked_tokens

        result.append(dists_model)

        model_kl = (total_kl / total_masked_tokens).item()
        model_perplexity = torch.exp(total_nll / total_masked_tokens).item()
        model_accuracy = (total_correct_predictions / total_masked_tokens).item()

        kl_values.append(model_kl)

        logger.info(f' KL(model[{i}], U) = {model_kl:.2f}bits')
        logger.info(f' PPL(model[{i}])   = {model_perplexity:.2f}')
        logger.info(f' ACC(model[{i}])   = {model_accuracy*100:.2f}%')

    logger.info(f' total masked tokens = {total_masked_tokens}')

    return torch.stack(result), kl_values

def top_p_reweight(distributions: Tensor, p: float) -> Tensor:
    '''Reweight distributions to keep only the top-p% probability mass.'''

    assert torch.allclose(distributions.sum(dim=-1), torch.ones(1, device=distributions.device)) 
    # should be already softmax-ed and sum to 1

    sorted_probs, sorted_idx = torch.sort(distributions, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > p
    mask[:, :, 0] = False # at least one token

    sorted_probs[mask] = 0.0

    total = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = sorted_probs / torch.clamp(total, min=1e-8)

    return torch.zeros_like(distributions).scatter_(-1, sorted_idx, sorted_probs)
