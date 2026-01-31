import torch
from tqdm import tqdm
from torch import Tensor
from logging import Logger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer
from scipy.spatial.distance import jensenshannon

from src.analyze.data import DeviceWrapper, mlm_preprocess, get_dataset_dna, get_dataset_text
from src.analyze import metrics as m
from src.analyze.distributions import top_p_reweight
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

        device = next(models[0].parameters()).device

        logger.info(f' run[{run}] n_models={len(models)}')

        is_text = 'text' in run

        attn_file = f'{run}_{n_samples}.attn'

        if cache.cached(attn_file): 
            attn_models = cache.get(attn_file)
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
            dataloader = DataLoader(DeviceWrapper(encoded, device=device), batch_size=batch_size)

            attn_models = get_attention_scores(models, dataloader)
            cache.store(attn_file, attn_models)

        ps: list = [0.05, 0.1, 0.2, 0.5, 0.9, 1.0]
        scores_all = {p: torch.zeros((12, 12)) for p in ps}

        total = 0

        for a in range(len(models)):
            for b in range(len(models))[a+1:]:
                print(a, b)
                scores = compare_models_attn(attn_models[a], attn_models[b], device, ps=ps)
                for p in ps: scores_all[p] += scores[p]
                total += 1
        
        for p in ps:
            scores_all[p] /= total
            scores_all[p] = (scores_all[p] + scores_all[p].T) / 2 

        data[run] = scores_all

    return data

def compare_models_attn(attnA, attnB, device, ps):
    scores_all = {p: torch.empty((12, 12)) for p in ps}

    for lA in attnA:
        for lB in attnB:

            layerA = attnA[lA]
            layerB = attnB[lB]

            a = torch.cat([layerA[head] for head in layerA], dim=0).to(device)
            b = torch.cat([layerB[head] for head in layerB], dim=0).to(device)

            for p in ps:

                a_top_p = top_p_reweight(a, p=p)
                b_top_p = top_p_reweight(b, p=p)

                a_unit = F.normalize(a_top_p, dim=1)
                b_unit = F.normalize(b_top_p, dim=1)

                score = (a_unit * b_unit).sum(dim=1).clamp(min=0.0).mean()

                scores_all[p][lA, lB] = score.item()

    return scores_all

@torch.autograd.inference_mode()
def get_attention_scores(
    models: tuple[BertForMaskedLM], 
    dataloader: DataLoader, 
)-> dict[int, dict[str, Tensor]]:
    '''
    Extract attention scores.

    Returns a dict of [models, layers, heads].
    '''

    attn = {}
    for i, model in enumerate(models):
        model.eval()

        attn_model = {}
        for batch in tqdm(dataloader, desc=f'model {i+1}/{len(models)}'):

            for layer, attn_layer in enumerate(model(**batch).attentions):
                if layer not in attn_model.keys(): attn_model[layer] = {}

                for head, attn_head in enumerate(attn_layer[0]):

                    if head not in attn_model[layer].keys(): attn_model[layer][head] = []

                    labels = batch['labels']
                    mask = (labels != -100)[0]
                    masked_attn = attn_head[mask]
                    attn_model[layer][head].append(masked_attn.detach())

        for layer in range(model.config.num_hidden_layers):
            for head in range(model.config.num_attention_heads):
                attn_model[layer][head] = torch.cat(attn_model[layer][head]).cpu()

        attn[i] = attn_model
        model.cpu()

    return attn
