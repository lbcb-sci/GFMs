import os
import torch
import argparse
from pathlib import Path
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer
import pprint

from .static  import static_analysis
from .dynamic import dynamic_analysis
from src.utils import get_logger, count_parameters

def load_tokenizer(args):
    runs = [Path(run) for run in args.runs]

    paths = {}
    for run in runs:
        tokenizers = os.listdir(run)
        paths[run] = [run / tokenizer for tokenizer in tokenizers]

    tokenizers = {}
    for run, models_path in paths.items():
        tokenizers[run] = []
        for model_path in models_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            return tokenizer

def load_models(args) -> dict:
    logger = args.logger
    device = args.device
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
            config = BertConfig.from_pretrained(model_path, local_files_only=True)
            config.output_attentions = True # output attention scores

            model = BertForMaskedLM.from_pretrained(model_path, config=config).to(device)

            logger.info(f' model [{model_path}] has {count_parameters(model):,} parameters')
            models[run].append(model)

    return models

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the trained models.')
    parser.add_argument('--runs', type=str, nargs='+', required=True)
    parser.add_argument('--type', type=str, required=True, choices=['static', 'dynamic'])
    return parser.parse_args()

@torch.autograd.inference_mode()
def main() -> None:
    import warnings
    warnings.simplefilter('ignore')

    args = parse_args()

    logger = get_logger('<analyze>')
    logger.info(f' args: {args}')
    args.logger = logger

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f' running computations on device {args.device}')

    models = load_models(args)
    models = {str(k).replace('runs/', ''): v for k, v in models.items()}

    match args.type:
        case 'static': 
            logger.info(' running static analysis...')
            static_results = static_analysis(models, logger)
            logger.info(' static analysis done.\n')

            for run, metrics in static_results.items():
                logger.info(' ' + run)
                for metric, (mean, std) in metrics.items():  
                    logger.info(f' run[{run}] metric[{metric}]: {mean:.3f} ({std:.5f})')

        case 'dynamic': 
            logger.info(' running dynamic analysis...')
            tokenizer = load_tokenizer(args)
            dynamic_results = dynamic_analysis(models, tokenizer, logger)

if __name__ == '__main__': main()
