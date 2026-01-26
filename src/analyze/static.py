import torch 
from torch import Tensor
from typing import Callable, Iterable
from ckatorch.core import cka_base
from transformers import BertForMaskedLM

def extract_word_embeddings(models: list[BertForMaskedLM]) -> Tensor:
    '''Returns a 3d tensor of stacked word embeddings.'''
    return torch.stack([model.bert.embeddings.word_embeddings.weight.detach() for model in models], dim=0)

def pairwise_metric(matrices: Iterable, metric: Callable) -> Tensor:
    '''Compute metric `metric` over all unique pairs of matrices.'''
    assert callable(metric)
    results = []
    for i, a in enumerate(matrices):
        for b in matrices[i+1:]: results.append(metric(a, b))
    return torch.as_tensor(results)

def meanstd(values: Iterable) -> tuple[float, float]:
    return torch.mean(values).item(), torch.std(values).item()

def cka(embeddings): return meanstd(pairwise_metric(embeddings, metric=cka_base))

def static_analysis(models_dict: dict) -> None:
    data = {run: {} for run in models_dict.keys()}

    for run, models in models_dict.items():
        embeddings = extract_word_embeddings(models)

        # CKA
        cka_result = cka(embeddings)
        data[run]['cka'] = cka_result

    return data
