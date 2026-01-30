import torch
from tqdm import tqdm
from torch import Tensor
from logging import Logger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer

from src.analyze.data import mlm_preprocess, get_dataset_dna, get_dataset_text, DeviceWrapper
from src.analyze import metrics
from src.utils import cache

def analyze_fisher(
    models_dict: dict,
    tokenizer: PreTrainedTokenizer,
    logger: Logger,
    n_samples: int,
    batch_size: int,
    p_mask: float = 0.15,
) -> dict:

    preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=p_mask)

    logger.info(f' computing fisher metrics')
    logger.info(f' tokenizer: {type(tokenizer)} from {tokenizer.name_or_path}')

    data = {run: {} for run in models_dict.keys()}

    for run, models in models_dict.items():
        models = [models[0]]

        logger.info(f' run[{run}] n_models={len(models)}')

        is_text = 'text' in run

        fisher_file = f'{run}_{n_samples}.fisher'

        if cache.cached(fisher_file): 
            fisher_params = cache.get(fisher_file)
            logger.info(' fisher data retrieved from cache')

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
            fisher_params = get_fisher_params(models, dataloader)
            cache.store(fisher_file, fisher_params)

        #shared = metrics.compute_pairwise(fisher, fisher_cosine)
        data[run]['encoder_dominance'] = encoder_dominance(fisher_params)
        data[run]['fisher'] = {model: reduce_fisher(x) for model, x in fisher_params.items()}
        for m in models: m.cpu()

    return data

def encoder_dominance(fisher):
    grouped = group_fisher(fisher)

    enc  = grouped['encoder']
    emb  = grouped['embeddings']
    head = grouped['head']

    results = []
    for en, em, h in zip(enc, emb, head):
        encoder_dominance = en / (en + em + h)
        results.append(encoder_dominance)

    return results

def flatten(fisher: dict[str, Tensor]):
    result = torch.cat([torch.flatten(t) for t in fisher.values()])
    return result

def fisher_cosine(fisherA: dict[str, Tensor], fisherB: dict[str, Tensor]) -> float:
    A = flatten(fisherA)
    B = flatten(fisherB)
    top = A @ B
    bot = A.norm() * B.norm()
    return (top / bot).item()

@torch.autograd.enable_grad()
def get_fisher_params(
    models: tuple[BertForMaskedLM],
    dataloader: DataLoader,
) -> dict[int, dict[str, Tensor]]:
    '''
    Compute averaged squared gradients of masked cross entropy loss = log likelihood.

    Empirical estimate of diag(Fisher Information Matrix(theta, D)).
    '''

    model = models[0]

    result = {}

    for i, model in enumerate(models):
        model.eval()

        fisher = {}
        total_tokens = 0

        for batch in tqdm(dataloader):
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch['labels']

            mask = labels != -100
            num_masked = mask.sum().item()
            total_tokens += num_masked

            masked_labels = labels[mask]
            ll = F.cross_entropy(logits[mask], masked_labels, reduction='sum')

            model.zero_grad(set_to_none=True)
            ll.backward()

            for n, p in model.named_parameters():
                if p.grad is None or p.numel() < 10_000:
                    continue

                if n not in fisher.keys():
                    fisher[n] = torch.zeros_like(p)

                fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] = (fisher[n] / total_tokens).cpu()

        result[i] = fisher

        model.cpu()

    return result

def reduce_fisher(fisher: dict):
    for layer, tensor in fisher.items():
        fisher[layer] = tensor.mean().item()
    return fisher

def group_fisher(fisher: dict[str, Tensor]):
    result = {'embeddings': [], 'encoder': [], 'head': []}

    for model in fisher.values():

        total_head = 0.0
        total_embeddings = 0.0
        total_encoder = 0.0

        for layer, tensor in model.items():
            if   'cls' in layer : total_head += tensor.sum().item()
            elif 'embeddings' in layer: total_embeddings += tensor.sum().item()
            elif 'encoder' in layer: total_encoder += tensor.sum().item()
            else: raise Exception(layer)

        result['embeddings'].append(total_embeddings)
        result['encoder'].append(total_encoder)
        result['head'].append(total_head)

    return result
