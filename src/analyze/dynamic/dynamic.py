import os
import numpy
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from transformers import BertForMaskedLM, PreTrainedTokenizer
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import get_cache_path, get_plots_path
from .data import get_dataset_dna, get_dataset_text, CudaWrapper

def mlm_preprocess(batch, tokenizer: PreTrainedTokenizer, mask_prob: float):
    texts = batch['text']
    tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    input_ids = tokenized['input_ids']
    mask_labels = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_arr = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)
    tokenized['input_ids'][mask_arr] = tokenizer.mask_token_id
    tokenized['labels'] = mask_labels
    return tokenized

@torch.autograd.inference_mode()
def get_masked_logits(
    models: list[BertForMaskedLM], 
    dataloader: DataLoader, 
    tokenizer: PreTrainedTokenizer,
)-> dict[int, Tensor]:
    '''Extract only masked token logits.'''

    logits = {}

    for i, model in enumerate(models):
        logits_i = []

        for batch in dataloader:
            logits_batch = model(**batch).logits

            masked_mask = (
                (batch['input_ids'] == tokenizer.mask_token_id) & 
                (batch['input_ids'] != tokenizer.pad_token_id)  & 
                batch['attention_mask'].bool()
            )

            masked_logits = logits_batch[masked_mask]
            logits_i.append(masked_logits)

        logits[i] = torch.cat(logits_i, dim=0)

    return logits

def top_p(probs: Tensor, p: float) -> Tensor:
    '''Reweight distribution to keep only the top-p mass.'''

    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=1)

    mask = cum_probs > p
    mask[:, 0] = False

    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=1, keepdim=True)

    out = torch.zeros_like(probs)
    out.scatter_(1, sorted_idx, sorted_probs)
    return out

def jensen_shannon(probs: Tensor) -> float:
    '''Pairwise Jensen-Shannon distance.'''

    result = []
    for i, a in enumerate(probs):
        for b in probs[i+1:]: 
            result.append(jensenshannon(a.cpu(), b.cpu(), base=2))
    return numpy.mean(result)

def analyze_logits(
    logits: dict[int, Tensor],
    p_values: list[float] = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
) -> dict:
    '''Analyze the output distributions (on masked tokens).'''

    js = {p: [] for p in p_values}
    kl_uniform = []

    n_tokens = logits[0].shape[0]
    vocab_size = logits[0].shape[1]
    
    uniform = torch.ones(vocab_size) / vocab_size

    for i in range(n_tokens):
        logits_token = torch.stack([logits[model][i] for model in logits.keys()])
        probs = F.softmax(logits_token, dim=1)

        # KL(Uniform, Probs) = suprisal induced by Props under a Uniform distribution
        kl_uniform.append(entropy(pk=probs.cpu(), qk=uniform, base=2, axis=1).mean())

        # Jensen-Shannon
        for p in p_values:
            probs_top_p = top_p(probs, p)
            js_p = jensen_shannon(probs_top_p)
            js[p].append(js_p)

    for k, v in js.items(): js[k] = numpy.mean(v)
    return {'js': js, 'kl_uniform': numpy.mean(kl_uniform)}

def dynamic_analysis(models_dict: dict, tokenizer: PreTrainedTokenizer, logger):
    preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.15)

    logger.info(f' tokenizer: {type(tokenizer)} from {tokenizer.name_or_path}')
    data = {}

    nsamples = 1024
    batch_size = 128

    cache_path = get_cache_path()

    for run, models in models_dict.items():
        data[run] = {}
        logits_file = f'{run}.logits'
        logger.info(f' run: {run}')

        if logits_file in os.listdir(get_cache_path()):
            logits = torch.load(cache_path / logits_file)
        else:
            if 'dna' in run:
                logger.info(f' collecting dataset dna...')
                dataset = get_dataset_dna(n=nsamples)
                remove = ['text']
        
            elif 'text' in run:
                logger.info(f' collecting dataset text...')
                dataset = get_dataset_text(n=nsamples)
                remove = ['text', 'url', 'id', 'title']

            encoded = dataset.map(preprocess, batched=True, remove_columns=remove)
            encoded.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            dataloader = DataLoader(CudaWrapper(encoded), batch_size=batch_size, num_workers=0)
            logits = get_masked_logits(models, dataloader, tokenizer)
            torch.save(logits, cache_path / logits_file)

        data[run] = analyze_logits(logits)

    print(data)

    for run, result in data.items():
        js = result['js']
        plt.plot(list(reversed(list(js.keys()))), list(reversed(list(js.values()))), label=run)
        plt.legend()
        plt.ylim((0.0, 1.0))

    plt.savefig(get_plots_path() / f'js.png', dpi=300)
    plt.close()
