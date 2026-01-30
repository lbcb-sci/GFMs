import numpy, torch
from torch import Tensor
from logging import Logger
from tqdm import tqdm, trange
from scipy.stats import entropy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer

from src.analyze.data import mlm_preprocess, get_dataset_dna, get_dataset_text, DeviceWrapper
from src.analyze import metrics as m
from src.utils import cache

def analyze_distributions(
    models_dict: dict, 
    tokenizer: PreTrainedTokenizer, 
    logger: Logger,
    n_samples: int,
    batch_size: int,
    p_mask: float = 0.15,
) -> dict:
    '''Analyze the token distributions of BERT models over masked tokens.'''

    preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=p_mask)

    logger.info(f' computing metrics on distributions')
    logger.info(f' tokenizer: {type(tokenizer)} from {tokenizer.name_or_path}')

    data = {run: {} for run in models_dict.keys()}

    for run, models in models_dict.items():

        logger.info(f' run[{run}] n_models={len(models)}')

        is_text = 'text' in run

        logits_file = f'{run}_{n_samples}.logits'

        if cache.cached(logits_file): 
            logits = cache.get(logits_file)
            logger.info(' logits retrieved from cache')

        else:

            if is_text:
                logger.info(f' collecting dataset text...')
                dataset = get_dataset_text(n=n_samples)
                remove = ['text', 'url', 'id', 'title']
            else:
                logger.info(f' collecting dataset dna...')
                dataset = get_dataset_dna(n=n_samples)
                remove = ['text']

            encoded = dataset.map(preprocess, batched=True, remove_columns=remove)
            encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            dataloader = DataLoader(DeviceWrapper(encoded, device=models[0].device), batch_size=batch_size)
            logits = get_masked_logits(models, dataloader, tokenizer)
            cache.store(logits_file, logits)

        perplexity = [m['perplexity'] for m in logits.values()]
        accuracy = [m['accuracy'] for m in logits.values()]
        logits = {i: m['logits'] for i, m in logits.items()}

        data[run] = compute_metrics(logits)
        data[run]['perplexity'] = perplexity
        data[run]['accuracy'] = accuracy

    return data

TOP_P_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def compute_metrics(
    logits: dict[int, Tensor],
    top_p_values: list[float] = TOP_P_VALUES,
) -> dict:
    '''
    Analyze the output distributions (on masked tokens).
    
    Computes: 
        - Mean output distribution.
        - Top-p Jensen-Shannon distance between models for different values of p.
        - KL divergence between models and the uniform distribution.
    '''

    kl_uniform = []
    jensen_shannon = {p: [] for p in top_p_values}
    n_tokens, vocab_size = logits[0].shape
    mean_dist = torch.zeros(vocab_size)
    uniform = torch.ones(vocab_size) / vocab_size

    for i in trange(0, n_tokens, 1, desc='computing metrics on logits'):

        # get probability distributions
        logits_token = torch.stack([logits[model][i] for model in logits.keys()])
        probs = F.softmax(logits_token, dim=1).cpu()

        # sort token proabilities
        sorted_probs, _ = torch.sort(probs, descending=True)
        mean_dist += sorted_probs.mean(dim=0)

        # KL(model, uniform)
        kl_uniform.append(entropy(pk=probs, qk=uniform, base=2, axis=1).mean())

        # Jensen-Shannon
        for p in top_p_values: jensen_shannon[p].append(m.jensen_shannon(top_p(probs, p)))

    for k, v in jensen_shannon.items(): jensen_shannon[k] = numpy.mean(v)
    kl_uniform = numpy.mean(kl_uniform)
    mean_dist = (mean_dist / n_tokens)

    return {'jensen_shannon': jensen_shannon, 'kl_uniform': kl_uniform, 'mean_dist': mean_dist}

@torch.autograd.inference_mode()
def get_masked_logits(
    models: tuple[BertForMaskedLM], 
    dataloader: DataLoader, 
    tokenizer: PreTrainedTokenizer
)-> dict[int, dict[str, Tensor]]:
    '''Extract masked token logits. Also computes perplexity and accuracy.'''

    logits = {}; results = {}

    for i, model in enumerate(models):
        model.eval()

        logits_i = []
        nll = correct_tokens = total_tokens = 0

        for batch in tqdm(dataloader, desc=f'model {i+1}/{len(models)}'):

            logits_batch = model(**batch).logits
            labels = batch['labels']
            mask = (
                (batch['input_ids'] == tokenizer.mask_token_id) & 
                (batch['input_ids'] != tokenizer.pad_token_id) & 
                batch['attention_mask'].bool()
            )

            masked_labels = labels[mask]
            masked_logits = logits_batch[mask]

            logits_i.append(masked_logits.detach())

            batch_nll = F.cross_entropy(masked_logits, masked_labels, reduction='mean')
            preds = masked_logits.argmax(dim=-1)
            correct_tokens += (preds == masked_labels).sum().item()
            total_tokens += masked_labels.numel()
            nll += batch_nll.item() * masked_labels.numel()

        logits[i]  = torch.cat(logits_i, dim=0)
        perplexity = torch.exp(torch.tensor(nll) / total_tokens).item()
        accuracy   = correct_tokens / total_tokens

        results[i] = {
            'logits': logits[i],
            'accuracy': accuracy,
            'perplexity': perplexity,
        }

    return results

def top_p(probs: Tensor, p: float) -> Tensor:
    '''Reweight distribution to keep only the top-p% probability mass.'''

    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, device=probs.device)) 
    # should be already softmax-ed and sum to 1

    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=1)

    mask = cum_probs > p
    mask[:, 0] = False # at least one token

    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=1, keepdim=True)

    return torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
