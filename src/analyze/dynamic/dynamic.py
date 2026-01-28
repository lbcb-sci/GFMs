import os
import numpy
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from transformers import BertForMaskedLM, PreTrainedTokenizer
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm, trange

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

    correct = 0
    total = 0

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

    print(f'Masked prediction accuracy: {correct.item() / total:.4f}')

    return logits

def top_p(probs: Tensor, p: float) -> Tensor:
    '''Reweight distribution to keep only the top-p% mass.'''

    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=1)

    mask = cum_probs > p
    mask[:, 0] = False

    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=1, keepdim=True)

    return torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

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
    
    mean_sorted_dist = torch.zeros(vocab_size, device=logits[0].device)
    uniform = torch.ones(vocab_size) / vocab_size

    for i in trange(0, n_tokens, 1, desc='computing js'):

        logits_token = torch.stack([logits[model][i] for model in logits.keys()])
        probs = F.softmax(logits_token, dim=1)

        sorted_probs, _ = torch.sort(probs, descending=True)
        mean_sorted_dist += sorted_probs.mean(dim=0)

        # KL(Uniform, Probs) = suprisal induced by Props under a Uniform distribution
        kl_uniform.append(entropy(pk=probs.cpu(), qk=uniform, base=2, axis=1).mean())

        # Jensen-Shannon
        for p in p_values:
            probs_top_p = top_p(probs, p)
            js_p = jensen_shannon(probs_top_p)
            js[p].append(js_p)

    mean_sorted_dist /= n_tokens

    for k, v in js.items(): js[k] = numpy.mean(v)
    return {'js': js, 'kl_uniform': numpy.mean(kl_uniform), 'mean_dist': mean_sorted_dist.cpu()}

def dynamic_analysis(models_dict: dict, tokenizer: PreTrainedTokenizer, logger):
    preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.15)

    #seq = 'AAAAAACCCCCCTTTTTTGGGGGG'
    #out = tokenizer(seq)['input_ids']
    #dec = tokenizer.decode(out)
    #print(seq)
    #print(out)
    #print(dec)
    #exit()

    logger.info(f' tokenizer: {type(tokenizer)} from {tokenizer.name_or_path}')
    data = {}

    nsamples = 128
    batch_size = 4

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
            encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            dataloader = DataLoader(CudaWrapper(encoded), batch_size=batch_size, num_workers=0)
            logits = get_masked_logits(models, dataloader, tokenizer)
            torch.save(logits, cache_path / logits_file)

        data[run] = analyze_logits(logits)

    pprint(data)

    for run, result in data.items():
        js = result['js']

        keys = numpy.array(list(js.keys()))
        values = numpy.array(list(js.values()))

        idx = numpy.argsort(keys)
        keys = numpy.sort(keys)
        values = values[idx]

        print(keys)
        print(values)

        plt.plot(keys, values, label=run)

    plt.ylim((0.0, 1.0))
    plt.title('Jensen-Shannon Distance as a Function of Top-P')
    plt.ylabel('JS')
    plt.xlabel('top-p mass kept')
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_plots_path() / f'js.png', dpi=300)
    plt.close()

    n = 20
    fig, ax = plt.subplots(3, figsize=(10, 10))
    for i, (run, values) in enumerate(data.items()):
        ax[i].bar(list(range(n)), values['mean_dist'][:n])
        ax[i].set_ylim((0.0, 1.0))
        ax[i].set_title(run)

    plt.tight_layout()
    plt.savefig(get_plots_path() / 'dists.png', dpi=300)
    plt.close()