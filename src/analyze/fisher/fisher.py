import torch
from tqdm import tqdm
from torch import Tensor
from logging import Logger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer

from src.analyze.data import mlm_preprocess, get_dataset_dna, get_dataset_text, DeviceWrapper

def analyze_fisher(models_dict: dict, tokenizers: list[PreTrainedTokenizer], args) -> dict:

    logger: Logger = args.logger
    n_samples: int = args.samples 
    batch_size: int = args.batch_size

    logger.info(f' computing fisher information...')

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
        preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.05)
        encoded = dataset.map(preprocess, batched=True, remove_columns=remove, load_from_cache_file=False)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dataloader = DataLoader(
            DeviceWrapper(encoded, device=models[0].device),
            batch_size=batch_size,
            shuffle=False,
        )

        fisher_information = get_fisher_information(models, dataloader)
        reduced = reduce_fisher_models(fisher_information)
        reduced = reduce_fisher_group(reduced)
        #reduced = reduce_fisher_normalize(reduced)
        reduced = reduce_fisher_group_more(reduced)
        reduced = reduce_fisher_sum(reduced)
        print(reduced)
        data[run] = reduced

        [model.cpu() for model in models]

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

@torch.autograd.enable_grad()
def get_fisher_information(
    models: tuple[BertForMaskedLM],
    dataloader: DataLoader,
    min_params_layer: int = 0,
) -> dict[int, dict[str, Tensor]]:
    '''
    Compute averaged squared gradients of negative log likelihood.

    Empirical estimate of diag(FIM).
    '''

    track_params = lambda params: \
        (params.grad is not None) and (params.numel() > min_params_layer)

    fisher_information = {}

    for i, model in enumerate(models):

        fisher_model = {}
        total_masked_tokens = 0

        for batch in tqdm(dataloader):
            labels = batch['labels']

            mask = (labels != -100)
            labels = labels[mask]

            num_masked_tokens = labels.numel()
            total_masked_tokens += num_masked_tokens

            logits = model(**batch).logits[mask]
            assert logits.shape[0] == num_masked_tokens

            negative_log_likelihood = F.cross_entropy(logits, labels)

            model.zero_grad()
            negative_log_likelihood.backward()

            for name, params in model.named_parameters():
                if not track_params(params): continue

                if name not in fisher_model: 
                    fisher_model[name] = torch.zeros_like(params.grad.flatten())

                fisher_model[name] += (params.grad**2).flatten()

        for name in fisher_model:
            fisher_model[name] /= num_masked_tokens

        fisher_information[i] = fisher_model

    return fisher_information

def reduce_fisher_sum(fisher: dict[str, Tensor]) -> dict[str, Tensor]:
    for layer, tensor in fisher.items():
        if isinstance(tensor, tuple): 
            fisher[layer] = tensor[0]
        else: fisher[layer] = tensor.sum().item()

    return fisher

def reduce_fisher_group_more(fisher: dict[str, Tensor]) -> dict[str, Tensor]:
    toremove = []
    sum_encoder = 0.0
    sum_params = 0
    for layer, (s, t) in fisher.items():
        if 'encoder' in layer:
            sum_encoder += s
            sum_params += t
            toremove.append(layer)

    for layer in toremove: fisher.pop(layer)
    fisher['encoder'] = (sum_encoder, sum_params)
    return fisher

def reduce_fisher_normalize(fisher: dict[str, Tensor]) -> dict[str, Tensor]:

    for layer, tensor in fisher.items():
        if isinstance(tensor, tuple): 
            fisher[layer] = tensor[0] / tensor[1]
        else: pass

    return fisher

def reduce_fisher_models(fisher: dict[int, dict]) -> dict[str, Tensor]:
    new_fisher = {}

    for model in fisher.values():
        for layer, tensor in model.items():
            if layer not in new_fisher:
                new_fisher[layer] = tensor
            else: new_fisher[layer] += tensor

    for layer, tensor in model.items():
        new_fisher[layer] /= len(fisher.keys())

    return new_fisher

def reduce_fisher_group(fisher: dict[str, Tensor], num_encoder_layers: int = 12) -> dict[str, Tensor]:
    match_layer = lambda i, layer: layer.startswith(f'bert.encoder.layer') and int(layer.split('.')[3]) == i

    sum_embeddings = 0.0
    params_embeddings = 0

    for layer, tensor in fisher.items():
        if not 'embeddings' in layer: continue
        params_embeddings += tensor.numel()
        sum_embeddings += tensor.sum().item()

    toremove = []
    for layer, tensor in fisher.items():
        if not 'embeddings' in layer: continue
        toremove.append(layer)

    for layer in toremove: fisher.pop(layer)

    fisher[f'embeddings'] = (sum_embeddings, params_embeddings)

    sum_head = 0.0
    params_head = 0

    for layer, tensor in fisher.items():
        if not 'cls' in layer: continue
        params_head += tensor.numel()
        sum_head += tensor.sum().item()

    toremove = []
    for layer, tensor in fisher.items():
        if not 'cls' in layer: continue
        toremove.append(layer)

    for layer in toremove: fisher.pop(layer)

    fisher[f'head'] = (sum_head, params_head)

    for i in range(num_encoder_layers):
        sum_encoder = 0.0
        params_encoder = 0

        for layer, tensor in fisher.items():
            if not match_layer(i, layer): continue
            params_encoder += tensor.numel()
            sum_encoder += tensor.sum().item()

        toremove = []
        for layer, tensor in fisher.items():
            if not match_layer(i, layer): continue
            toremove.append(layer)

        for layer in toremove: fisher.pop(layer)

        fisher[f'encoder.{i}'] = (sum_encoder, params_encoder)

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
