import os
import argparse
from pathlib import Path
from pprint import pprint
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer
import matplotlib.pyplot as plt

from src.utils import get_logger, count_parameters

from src.analyze.distributions import analyze_distributions
from src.analyze.static import analyze_word_embeddings
from src.analyze.fisher import analyze_fisher
from src.analyze.attention import analyze_attention

def main() -> None:
    args = parse_args()

    logger = get_logger('<analyze>')
    args.logger = logger
    logger.info(f' args: {args}')

    logger.info(f' running on device {args.device}')

    models = load_models(args)
    models = {str(k).replace('runs/', ''): v for k, v in models.items()}

    match args.type:

        case 'static': 
            logger.info(' running word-embeddings analysis...')
            results = analyze_word_embeddings(models, logger)
            logger.info(' word-embeddings analysis done.\n')
            pprint(results)

        case 'distributions': 
            logger.info(' running distributions analysis...')
            tokenizer = load_tokenizer(args)
            results = analyze_distributions(
                models, tokenizer, logger,
                n_samples=args.samples,
                batch_size=args.batch_size,
            )
            logger.info(' distributions analysis done\n')
            pprint(results)

        case 'fisher':
            logger.info(' running fisher analysis...')
            tokenizer = load_tokenizer(args)
            results = analyze_fisher(
                models, tokenizer, logger,
                n_samples=args.samples,
                batch_size=args.batch_size,
            )
            logger.info(' fisher analysis done\n')
            #pprint(results)

            for run in results.keys():

                print(results[run]['encoder_dominance'])

                max_layer = ''
                maxval = float('-inf')
                for layer, val in results[run]['fisher'][0].items(): 
                    if val > maxval:
                        max_layer = layer
                        maxval = val

                print(max_layer)
                print(maxval)

                fig, ax = plt.subplots(4, figsize=(30, 15))
                for i, model in enumerate(results[run]['fisher'].values()):
                    ax[i].bar(list(model.keys()), list(model.values()), color='red' if 'dna' in run else 'blue')
                    ax[i].tick_params(axis='y', left=False, labelleft=False)
                    ax[i].tick_params(rotation=45)
                    if i < 3: ax[i].tick_params(axis='x', labelbottom=False)

                fig.suptitle(run)

                plt.tight_layout()
                plt.savefig(f'{run}.png', dpi=400)
                plt.close()

        case 'attention':
            logger.info(' running attention analysis...')
            tokenizer = load_tokenizer(args)
            results = analyze_attention(
                models, tokenizer, logger,
                n_samples=args.samples,
                batch_size=args.batch_size,
            )
            logger.info(' attention analysis done\n')
            pprint(results)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the trained models.')
    parser.add_argument('--type', type=str, required=True, 
                        choices=['static', 'distributions', 'fisher', 'attention'])
    parser.add_argument('--runs', type=str, nargs='+', required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def load_tokenizer(args):
    run = Path(args.runs[0])
    tokenizers = os.listdir(run)
    path = run / tokenizers[0]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
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

            model = BertForMaskedLM.from_pretrained(model_path, config=config, device_map=device)

            logger.info(f' model [{model_path}] has {count_parameters(model):,} parameters')
            models[run].append(model)

    return models

if __name__ == '__main__': main()
