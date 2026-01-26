import torch 
from torch import Tensor
from transformers import BertForMaskedLM

from .metrics import cosine_similarity, topk, cka

def extract_word_embeddings(models: list[BertForMaskedLM]) -> Tensor:
    '''Returns a 3d tensor of stacked word embeddings.'''
    return torch.stack([model.bert.embeddings.word_embeddings.weight.detach() for model in models], dim=0)

def static_analysis(models_dict: dict, logger) -> dict:
    '''Main static analysis function.'''

    data = {run: {} for run in models_dict.keys()}

    for run, models in models_dict.items():
        logger.info(f' extracting word embeddings for run {run}...')
        embeddings = extract_word_embeddings(models)
        logger.info(f' extracting word embeddings done.')

        logger.info(f' computing cosine similarities...')
        cosims = torch.stack([cosine_similarity(E, E) for E in embeddings], dim=0)
        logger.info(f' computing cosine similarities done.')

        #### TOP-K
        logger.info(f' computing top-3 overlap...')
        data[run]['top3']  = topk(cosims, k=3)

        logger.info(f' computing top-10 overlap...')
        data[run]['top10'] = topk(cosims, k=10)

        logger.info(f' computing top-k overlap done.')

        #### CKA
        logger.info(f' computing cka...')
        data[run]['cka'] = cka(embeddings)

        logger.info(f' computing cka done.')

    return data
