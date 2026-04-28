import torch
from tqdm import tqdm
from torch import Tensor
from logging import Logger
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

from src.analyze.data import mlm_preprocess, get_dna_eval_dataset, get_wikipedia, DeviceWrapper
from src.utils import create_results_dict, DATA_TOKENIZER_PAIRS

def fisher(all_models: dict, tokenizers: dict, args) -> dict:
    logger, n_samples, batch_size = args.logger, args.samples, args.batch_size

    logger.info(f' computing fisher information...')

    results = create_results_dict()

    for data, tok in DATA_TOKENIZER_PAIRS:
        models = all_models[data][tok]
        tokenizer = tokenizers[data][tok][0]

        assert all([model.name_or_path[:-1] == tokenizer.name_or_path[:-1] for model in models])

        is_text = 'text' in tokenizer.name_or_path

        logger.info(f' collecting dataset {"text" if is_text else "dna"}...')
        dataset = get_wikipedia(n_samples) if is_text else get_dna_eval_dataset(n_samples, args.dna_dataset_path)
        remove = ['text', 'url', 'id', 'title'] if is_text else ['text']

        logger.info( 'masking tokens in dataset...')
        preprocess = lambda batch: mlm_preprocess(batch, tokenizer, mask_prob=0.15)
        encoded = dataset.map(preprocess, batched=True, remove_columns=remove, load_from_cache_file=False)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        dataloader = DataLoader(
            DeviceWrapper(encoded, device=models[0].device),
            batch_size=batch_size,
            shuffle=False,
        )

        fisher_information = get_fisher_information(models, dataloader, logger)

        results[data][tok]['fisher'] = reduce_fisher_average_models(fisher_information)
        results[data][tok][f'fisher_full'] = reduce_fisher(fisher_information, collapse_encoder=False)

        [model.cpu() for model in models]

    return results

@torch.autograd.enable_grad()
def get_fisher_information(
    models: tuple[BertForMaskedLM],
    dataloader: DataLoader,
    logger,
) -> dict[int, dict[str, Tensor]]:
    '''Compute averaged squared gradients of negative log likelihood. Empirical estimate of diag(FIM).'''

    track_params = lambda params: params.grad is not None

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
            fisher_model[name] /= total_masked_tokens

        fisher_information[i] = fisher_model

        logger.info(f' total masked tokens = {total_masked_tokens}')

    return fisher_information

def reduce_fisher_average_models(
    fisher,
    *,
    group_layers=True,
    collapse_encoder=True,
    sum_only=True,
):
    out = {}
    for m in fisher.values():
        for k, v in m.items():
            out[k] = v.clone() if k not in out else out[k] + v
    for k in out:
        out[k] /= len(fisher)
    fisher = out

    # group layers
    if group_layers:
        grouped = defaultdict(list)

        def enc_idx(name):
            if name.startswith("bert.encoder.layer"):
                return int(name.split(".")[3])
            return None

        for name, tensor in fisher.items():
            if 'embeddings' in name: grouped['embeddings'].append(tensor)
            elif 'cls' in name: grouped['head'].append(tensor)
            else:
                i = enc_idx(name)
                if i is not None: grouped[f'encoder.{i}'].append(tensor)

        fisher = {
            k: (
                sum(t.sum().item() for t in v),
                sum(t.numel() for t in v),
            )
            for k, v in grouped.items()
        }

    # collapse encoders
    if collapse_encoder:
        enc_sum = 0.0
        enc_params = 0
        out = {}

        for k, (s, p) in fisher.items():
            if k.startswith('encoder.'):
                enc_sum += s
                enc_params += p
            else:
                out[k] = (s, p)

        if enc_params > 0:
            out['encoder'] = (enc_sum, enc_params)

        fisher = out

    # scalar output
    if sum_only: fisher = {k: v[0] for k, v in fisher.items()}

    return fisher

def reduce_fisher(
    fisher_models,
    *,
    group_layers=True,
    collapse_encoder=True,
    sum_only=True,
):
    result = {}

    for model in fisher_models:
        fisher = fisher_models[model]

        if group_layers:
            grouped = defaultdict(list)

            def enc_idx(name):
                if name.startswith("bert.encoder.layer"):
                    return int(name.split(".")[3])
                return None

            for name, tensor in fisher.items():
                if 'embeddings' in name: grouped['embeddings'].append(tensor)
                elif 'cls' in name: grouped['head'].append(tensor)
                else:
                    i = enc_idx(name)
                    if i is not None: grouped[f'encoder.{i}'].append(tensor)

            fisher = {
                k: (
                    sum(t.sum().item() for t in v),
                    sum(t.numel() for t in v),
                )
                for k, v in grouped.items()
            }

        if collapse_encoder:
            enc_sum = 0.0
            enc_params = 0
            out = {}

            for k, (s, p) in fisher.items():
                if k.startswith('encoder.'):
                    enc_sum += s
                    enc_params += p
                else:
                    out[k] = (s, p)

            if enc_params > 0:
                out['encoder'] = (enc_sum, enc_params)

            fisher = out

        # scalar output
        if sum_only: fisher = {k: v[0] for k, v in fisher.items()}
        result[model] = fisher

    return result
