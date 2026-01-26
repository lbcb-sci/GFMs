import os
import torch
import argparse
from pathlib import Path
from transformers import BertForMaskedLM

from src.utils import get_logger, count_parameters
from .static import static_analysis

def load_models(args) -> dict:
    logger = args.logger
    runs = [Path(run) for run in args.runs]

    paths = {}
    for run in runs:
        models = os.listdir(run)
        nmodels = len(models)
        logger.info(f' run [{run}] has {nmodels} models')
        paths[run] = [run / model for model in models]

    models = {}
    for run, models_path in paths.items():
        models[run] = []
        for model_path in models_path:
            model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
            logger.info(f' model [{model_path}] has {count_parameters(model):,} parameters')
            models[run].append(model)

    return models

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the trained models.')
    parser.add_argument('--runs', type=str, nargs='+', required=True)
    return parser.parse_args()

@torch.no_grad()
@torch.autograd.inference_mode()
def main() -> None:
    args = parse_args()

    logger = get_logger('<analyze>')
    logger.info(f' args: {args}')
    args.logger = logger

    models = load_models(args)

    logger.info(' running static analysis of word embeddings')
    static_analysis(models)

if __name__ == '__main__': main()
