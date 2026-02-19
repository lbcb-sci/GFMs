import os
import argparse
from pathlib import Path
from pprint import pprint
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer

from src.analyze.distribution import analyze_distributions
from src.analyze.static import analyze_word_embeddings
from src.analyze.fisher import analyze_fisher
from src.analyze.plotting import *
from src.utils import get_logger, count_parameters

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

            for run_name, run in results.items():
                print(run_name)
                for metric, (mean, std) in run.items():
                    print(f'- {metric}: {mean:.2f} {std:.2f}')
                print()

        case 'fisher': 

            logger.info(' running fisher information analysis...')
            tokenizers = load_tokenizers(args)
            results = analyze_fisher(models, tokenizers, args)
            logger.info(' fisher information analysis done.\n')
            pprint(results)

            dna_bpe_key = list(filter(lambda k: 'dna' in k and 'bpe' in k, list(models.keys())))[0]
            dna_kmer_key = list(filter(lambda k: 'dna' in k and 'kmer' in k, list(models.keys())))[0]
            text_key = list(filter(lambda k: 'text' in k, list(models.keys())))[0]

            text = results[text_key]['fisher']
            dna_bpe = results[dna_bpe_key]['fisher']
            dna_kmer = results[dna_kmer_key]['fisher']

            plot_fisher_information(text, dna_bpe, dna_kmer)

            text_full = results[text_key]['fisher_full']
            dna_bpe_full = results[dna_bpe_key]['fisher_full']
            dna_kmer_full = results[dna_kmer_key]['fisher_full']

            plot_full_fisher_information(text_full, dna_bpe_full, dna_kmer_full)

        case 'distribution': 

            logger.info(' running distributions analysis...')
            tokenizers = load_tokenizers(args)
            results = analyze_distributions(models, tokenizers, args)
            logger.info(' distributions analysis done\n')
            pprint(results)

            text_js     = results['90M_text_bpe']['js']
            dna_bpe_js  = results['90M_dna_bpe']['js']
            dna_kmer_js = results['90M_dna_kmer']['js']
            plot_jensen_shannon(text_js, dna_bpe_js, dna_kmer_js)

            n_text = 10
            n_dna = 50

            text_mean_dist = results['90M_text_bpe']['mean_dist'][:n_text]
            dna_bpe_mean_dist = results['90M_dna_bpe']['mean_dist'][:n_dna]
            dna_kmer_mean_dist = results['90M_dna_kmer']['mean_dist'][:n_dna]
            plot_average_distribution(text_mean_dist, dna_bpe_mean_dist, dna_kmer_mean_dist)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the trained models.')
    parser.add_argument('--type', type=str, required=True, choices=['static', 'distribution', 'fisher'])
    parser.add_argument('--runs', type=str, nargs='+', required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def load_tokenizers(args):
    tokenizers = []
    for run in args.runs:
        run = Path(run)
        path = run / os.listdir(run)[0]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizers.append(tokenizer)
    return tokenizers

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
            config.output_hidden_states = True # output hidden states

            model = BertForMaskedLM.from_pretrained(model_path, config=config, device_map=device).eval()

            logger.info(f' model [{model_path}] has {count_parameters(model):,} parameters')
            models[run].append(model)

    return models

if __name__ == '__main__': main()
