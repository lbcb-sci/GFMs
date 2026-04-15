import torch 
from torch import Tensor
import torch.nn.functional as F
from transformers import BertForMaskedLM

from src.analyze import metrics as m
from src.utils import DATA_TOKENIZER_PAIRS, N, create_results_dict

def word_embeddings(all_models: dict, logger) -> dict:
    '''Compute agreement metrics on static word embeddings.'''

    results = create_results_dict()
    
    K_VALUES = [1, 3, 5, 10, 20, 50, 100, 1000, 4000]

    for data, tok in DATA_TOKENIZER_PAIRS:

        logger.info(f' extracting word embeddings for run {data}-{tok}...')

        models = all_models[data][tok]
        embeddings = extract_word_embeddings(models)

        logger.info(f' computing cosine similarities...')
        cosine_similarities = torch.stack([m.cosine_similarity(E, E) for E in embeddings], dim=0)

        for k in K_VALUES:
            logger.info(f' computing top-{k} overlap...')
            results[data][tok][f'top_{k}'] = m.topk(cosine_similarities.cpu(), k=k)

            logger.info(f' computing local spearman k={k}...')
            results[data][tok][f'local_spearman_{k}'] = m.local_spearman(cosine_similarities.cpu(), k=k)

        logger.info(f' computing procrustes...')
        procrustes = m.get_procrustes(embeddings.cpu())

        disparities  = [disparity for (_, _, disparity) in procrustes]
        procrustes_X = [torch.tensor(X, device='cuda') for (X, _, _) in procrustes]
        procrustes_Y = [torch.tensor(Y, device='cuda') for (_, Y, _) in procrustes]

        procrustes_cosine_similarities = []
        for X, Y in zip(procrustes_X, procrustes_Y):
            procrustes_cosine_similarities.append((F.normalize(X) * F.normalize(Y)).sum(dim=1).mean().item())

        results[data][tok]['disparities'] = m.meanstd(disparities)
        results[data][tok]['procrustes_cosine'] = m.meanstd(procrustes_cosine_similarities)

        logger.info(f' computing linear cka...')
        results[data][tok]['linear_cka'] = m.cka(embeddings, kernel='linear')

        logger.info(f' computing rbf cka...')
        results[data][tok]['rbf_cka'] = m.cka(embeddings, kernel='rbf')

    return results

def extract_word_embeddings(models: list[BertForMaskedLM]) -> Tensor:
    '''Returns a 3d tensor of stacked word embeddings.'''
    return torch.stack([model.bert.embeddings.word_embeddings.weight.detach() for model in models], dim=0)
