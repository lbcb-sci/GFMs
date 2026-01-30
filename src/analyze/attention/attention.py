import torch
from tqdm import tqdm
from torch import Tensor
from logging import Logger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer

from src.analyze.data import DeviceWrapper, mlm_preprocess, get_dataset_dna, get_dataset_text
from src.utils import cache

def analyze_attention(
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

        attn_file = f'{run}_{n_samples}.attn'

        if cache.cached(attn_file): 
            logits = cache.get(attn_file)
            logger.info(' attention scores retrieved from cache')

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

            cache.store(logits_file, logits)

        perplexity = [m['perplexity'] for m in logits.values()]
        accuracy = [m['accuracy'] for m in logits.values()]
        logits = {i: m['logits'] for i, m in logits.items()}

        data[run] = compute_metrics(logits)
        data[run]['perplexity'] = perplexity
        data[run]['accuracy'] = accuracy

    return data

@torch.autograd.inference_mode()
def get_attention_scores(
    models: tuple[BertForMaskedLM], 
    dataloader: DataLoader, 
)-> dict[int, dict[str, Tensor]]:
    '''Extract masked token logits. Also computes perplexity and accuracy.'''

    attn = {}
    for i, model in enumerate(models):
        model.eval()
        attn_model = []

        for batch in tqdm(dataloader, desc=f'model {i+1}/{len(models)}'):
            attention_batch = model(**batch).attentions
            print(attention_batch)
            exit()

            labels = batch['labels']
            mask = batch['labels'] != -100

            masked_labels = labels[mask]
            masked_attn = attention_batch[mask]

            attn_model.append(masked_attn.detach())

        attn[i] = attn_model

    return attn