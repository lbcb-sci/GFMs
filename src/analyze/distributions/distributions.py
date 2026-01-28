import numpy, torch
from torch import Tensor
from logging import Logger
from tqdm import tqdm, trange
from scipy.stats import entropy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer

from .data import mlm_preprocess, get_dataset_dna, get_dataset_text, DeviceWrapper
from src.analyze import metrics as m
from src.utils import cache

def analyze_distributions(
    models_dict: dict, 
    tokenizer: PreTrainedTokenizer, 
    logger: Logger,
    n_samples: int = 512,
    batch_size: int = 4,
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
        
            logits = get_masked_logits(tuple(models), encoded, tokenizer, batch_size)
            cache.store(logits_file, logits)

        accuracy = logits['accuracy']
        logits = logits['logits']

        data[run] = compute_metrics(logits)
        data[run]['accuracy'] = accuracy

    return data

def compute_metrics(
    logits: dict[int, Tensor],
    top_p_values: list[float] = numpy.linspace(0.05, 1.0, num=5),
) -> dict:
    '''
    Analyze the output distributions (on masked tokens).
    
    Computes: 
        - Mean output distribution.
        - Top-p Jensen-Shannon distance between models for different values of p.
        - KL divergence between models and the uniform distribution.
    '''

    jensen_shannon = {p: [] for p in top_p_values}
    kldiv_uniform = []
    n_tokens, vocab_size = logits[0].shape
    mean_sorted_dist = torch.zeros(vocab_size, device=logits[0].device)
    uniform = torch.ones(vocab_size) / vocab_size

    for i in trange(0, n_tokens, 1, desc='computing metrics on logits'):

        # get probability distributions
        logits_token = torch.stack([logits[model][i] for model in logits.keys()])
        probs = F.softmax(logits_token, dim=1)

        # sort token proabilities
        sorted_probs, _ = torch.sort(probs, descending=True)
        mean_sorted_dist += sorted_probs.mean(dim=0)

        # KL(model, uniform)
        kldiv_uniform.append(entropy(pk=probs.cpu(), qk=uniform, base=2, axis=1).mean())

        # Jensen-Shannon
        for p in top_p_values: jensen_shannon[p].append(m.jensen_shannon(top_p(probs, p)))

    for k, v in jensen_shannon.items(): jensen_shannon[k] = numpy.mean(v)

    return {
        'jensen_shannon': jensen_shannon, 
        'kl_uniform': numpy.mean(kldiv_uniform), 
        'mean_dist': (mean_sorted_dist / n_tokens).cpu(),
    }

@torch.autograd.inference_mode()
def get_masked_logits(
    models: tuple[BertForMaskedLM], 
    dataset_encoded, 
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
)-> dict[int, Tensor]:

    '''Extract only masked token logits. Compute accuracy at the same time.'''

    dataloader = DataLoader(
        DeviceWrapper(dataset_encoded, device=models[0].device), 
        batch_size=batch_size, 
        num_workers=0,
    )

    logits = {}; total = correct = 0

    for i, model in enumerate(models):
        logits_i = []

        for batch in tqdm(dataloader, desc=f'getting logits for model {i}'):
            logits_batch = model(**batch).logits

            labels = batch['labels']

            masked_mask = (
                (batch['input_ids'] == tokenizer.mask_token_id) & 
                (batch['input_ids'] != tokenizer.pad_token_id)  & 
                batch['attention_mask'].bool()
            )

            masked_labels = labels[masked_mask]
            masked_logits = logits_batch[masked_mask]
            logits_i.append(masked_logits)

            correct += (masked_logits.argmax(dim=-1) == masked_labels).float().mean()
            total += 1

        logits[i] = torch.cat(logits_i, dim=0)

    return {'logits': logits, 'accuracy': correct.item() / total}

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
