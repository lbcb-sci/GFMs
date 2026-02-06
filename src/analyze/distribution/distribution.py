import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer

from src.analyze.data import mlm_preprocess, get_dataset_dna, get_dataset_text, DeviceWrapper
from src.analyze.metrics import kl_divergence, jensen_shannon_distance

def analyze_distributions(
    models_dict: dict, 
    tokenizers: list[PreTrainedTokenizer], 
    args,
) -> dict:

    logger = args.logger
    n_samples = args.samples
    batch_size = args.batch_size

    top_p_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    logger.info(f' computing metrics on distributions...')

    data = {run: {} for run in models_dict}

    for (run, models), tokenizer in zip(models_dict.items(), tokenizers):

        assert run in tokenizer.name_or_path

        logger.info(f' run[{run}] n_models={len(models)}')
        logger.info(f' tokenizer: {type(tokenizer)} from {tokenizer.name_or_path}')
        is_text = 'text' in run

        logger.info(f' collecting dataset {"text" if is_text else "dna"}...')
        dataset = get_dataset_text(n_samples) if is_text else get_dataset_dna(n_samples)
        remove = ['text', 'url', 'id', 'title'] if is_text else ['text']

        logger.info( 'masking tokens in dataset...')
        preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.10)
        encoded = dataset.map(preprocess, batched=True, remove_columns=remove, load_from_cache_file=False)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dataloader = DataLoader(
            DeviceWrapper(encoded, device=models[0].device),
            batch_size=batch_size,
            shuffle=False,
        )

        distributions = get_distributions(models, dataloader, logger)
        assert (distributions.sum(dim=-1) - 1 < 1e-6).all()

        mean_dist = compute_mean_distribution(distributions)
        data[run]['mean_dist'] = mean_dist.cpu()

        jensen_shannon = compute_jensen_shannon(distributions, top_p_values, logger)
        data[run]['js'] = jensen_shannon

        [model.cpu() for model in models]

    return data

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
)-> Tensor:
    '''Extract distributions over masked tokens. Also computes KL div, perplexity and accuracy.'''

    result = []

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
            assert (distributions.sum(dim=1) - 1 < 1e-6).all()

            uniform = torch.ones_like(distributions) / distributions.shape[-1]
            total_kl += kl_divergence(distributions, uniform).sum()

            dists_model.extend(distributions)

            total_correct_predictions += (distributions.argmax(-1) == labels).float().sum()

        dists_model = torch.stack(dists_model)
        assert dists_model.shape[0] == total_masked_tokens

        result.append(dists_model)

        model_kl = (total_kl / total_masked_tokens).item()
        model_perplexity = torch.exp(total_nll / total_masked_tokens).item()
        model_accuracy = (total_correct_predictions / total_masked_tokens).item()

        logger.info(f' KL(model[{i}], U) = {model_kl:.2f}bits')
        logger.info(f' PPL(model[{i}])   = {model_perplexity:.2f}')
        logger.info(f' ACC(model[{i}])   = {model_accuracy*100:.2f}%')

    logger.info(f' total masked tokens = {total_masked_tokens}')

    return torch.stack(result)

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
