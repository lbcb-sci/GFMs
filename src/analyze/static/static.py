import torch 
from torch import Tensor
from transformers import BertForMaskedLM

from src.analyze import metrics as m

def analyze_word_embeddings(models_dict: dict, logger) -> dict:
    '''Main static analysis function, collecting metrics and returning dict of values.'''

    data = {run: {} for run in models_dict.keys()}

    for run, models in models_dict.items():
        logger.info(f' extracting word embeddings for run {run}...')
        embeddings = extract_word_embeddings(models)
        logger.info(f' extracting word embeddings done.')

        logger.info(f' computing cosine similarities...')
        cosims = torch.stack([m.cosine_similarity(E, E) for E in embeddings], dim=0)
        logger.info(f' computing cosine similarities done.')

        #### TOP-K
        logger.info(f' computing top-3 overlap...')
        data[run]['top3'] = m.topk(cosims, k=3)

        logger.info(f' computing top-10 overlap...')
        data[run]['top10'] = m.topk(cosims, k=10)

        logger.info(f' computing top-k overlap done.')

        #### CKA
        logger.info(f' computing cka...')
        data[run]['linear cka'] = m.cka(embeddings, kernel='linear')
        data[run]['rbf cka'] = m.cka(embeddings, kernel='rbf')

        logger.info(f' computing cka done.')

        #### STD PER TOKEN
        logger.info(f' computing mean std(cosine_sim) per token.')
        data[run]['std'] = m.std_per_token(cosims)

    return data

def extract_word_embeddings(models: list[BertForMaskedLM]) -> Tensor:
    '''Returns a 3d tensor of stacked word embeddings.'''
    return torch.stack([model.bert.embeddings.word_embeddings.weight.detach() for model in models], dim=0)
