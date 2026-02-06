import torch 
from torch import Tensor
import torch.nn.functional as F
from transformers import BertForMaskedLM

from src.analyze import metrics as m

def analyze_word_embeddings(models_dict: dict, logger) -> dict:
    '''Main static analysis function, collecting metrics and returning dict of values.'''

    data = {run: {} for run in models_dict.keys()}
    
    k_values = [1, 3, 5, 10, 20, 50, 100, 1000, 4000]

    for run, models in models_dict.items():
        logger.info(f' extracting word embeddings for run {run}...')
        embeddings = extract_word_embeddings(models)

        logger.info(f' computing cosine similarities...')
        cosine_similarities = torch.stack([m.cosine_similarity(E, E) for E in embeddings], dim=0)

        for k in k_values:
            logger.info(f' computing top-{k} overlap...')
            data[run][f'top_{k}'] = m.topk(cosine_similarities.cpu(), k=k)

            logger.info(f' computing local spearman k={k}...')
            data[run][f'local_spearman_{k}'] = m.local_spearman(cosine_similarities.cpu(), k=k)

        logger.info(f' computing procrustes...')
        procrustes = m.get_procrustes(embeddings.cpu())

        disparities  = [disparity for (_, _, disparity) in procrustes]
        procrustes_X = [torch.tensor(X, device='cuda') for (X, _, _) in procrustes]
        procrustes_Y = [torch.tensor(Y, device='cuda') for (_, Y, _) in procrustes]

        procrustes_cosine_similarities = []
        for X, Y in zip(procrustes_X, procrustes_Y):
            procrustes_cosine_similarities.append((F.normalize(X) * F.normalize(Y)).sum(dim=1).mean().item())

        data[run]['disparities'] = m.meanstd(disparities)
        data[run]['procrustes_cosine'] = m.meanstd(procrustes_cosine_similarities)

    return data

def extract_word_embeddings(models: list[BertForMaskedLM]) -> Tensor:
    '''Returns a 3d tensor of stacked word embeddings.'''
    return torch.stack([model.bert.embeddings.word_embeddings.weight.detach() for model in models], dim=0)
