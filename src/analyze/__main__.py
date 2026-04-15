import argparse
from pathlib import Path
from pprint import pprint
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer

import src.analyze as analyze
from src.analyze.plotting import *
from src.utils import get_logger, count_parameters, DATA_TOKENIZER_PAIRS


def main() -> None:
    args = parse_args()

    logger = get_logger('<analyze>'); args.logger = logger

    logger.info(f' args: {args}')
    logger.info(f' running on device {args.device}')

    # paths = get_huggingface_paths()
    paths = get_local_paths(shuffle=False)
    pprint(paths)

    tokenizers = load_tokenizers(paths)
    models = load_models(paths, args.device)

    check_models(models, args.logger)

    match args.type:

        case 'static': 
            logger.info(' running word-embeddings analysis...')
            results = analyze.word_embeddings(models, logger)
            logger.info(' word-embeddings analysis done.\n')
            pprint(results)
            print_results(results)

        case 'fisher': 
            logger.info(' running fisher information analysis...')
            results = analyze.fisher(models, tokenizers, args)
            logger.info(' fisher information analysis done.\n')
            pprint(results)
            plot_fisher(results)

        case 'distribution': 
            logger.info(' running distributions analysis...')
            results = analyze.distributions(models, tokenizers, args)
            logger.info(' distributions analysis done\n')
            pprint(results)
            plot_distributions(results)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the trained models.')

    parser.add_argument('--type', type=str, required=True, choices=['static', 'distribution', 'fisher'])
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)

    return parser.parse_args()


def get_huggingface_paths() -> list[str]:
    N = 5; username = 'mrochk'
    result = {'text': {'bpe': []}, 'dna': {'bpe': [], 'kmer': []}}

    for data, tok in DATA_TOKENIZER_PAIRS:
        for idx in range(1, N+1):
            path = f'{username}/bert-90M-{data}-{tok}-{idx}'
            result[data][tok].append(path)

    return result


def get_local_paths(shuffle: bool) -> dict:
    N = 5
    base = Path('/home/vrcekl/scratch/GFMs/analyze')
    suffix = 'shuffle' if shuffle else 'nonshuffle'
    result = {'text': {'bpe': []}, 'dna': {'bpe': [], 'kmer': []}}

    for data, tok in DATA_TOKENIZER_PAIRS:
        for idx in range(1, N + 1):
            result[data][tok].append(base / f'{data}_{tok}_{suffix}' / str(idx))

    return result


def print_results(results: dict) -> None:
    for data, tok in DATA_TOKENIZER_PAIRS:
        print(data, tok)
        print('-' * 20)
        for metric, value in results[data][tok].items():
            print(f'{metric}\t\t{value}')


def load_tokenizers(paths: dict) -> dict:
    return {
        'text': {'bpe': [AutoTokenizer.from_pretrained(path) for path in paths['text']['bpe']]},
        'dna': {
            'bpe': [AutoTokenizer.from_pretrained(path) for path in paths['dna']['bpe']],
            'kmer': [AutoTokenizer.from_pretrained(path) for path in paths['dna']['kmer']],
        }
    }


def load_models(paths: dict, device) -> dict:
    models = {'text': {'bpe': []}, 'dna': {'bpe': [], 'kmer': []}}

    for data, tok in DATA_TOKENIZER_PAIRS:
        for path in paths[data][tok]:
            config = BertConfig.from_pretrained(path)
            config.output_attentions = True 
            config.output_hidden_states = True
            model = BertForMaskedLM.from_pretrained(path, config=config, device_map=device).eval()
            models[data][tok].append(model)

    return models


def check_models(models: dict, logger) -> None:
    for model in models['text']['bpe']:
        logger.info(f' model [{model.name_or_path}] has {count_parameters(model):,} parameters')

    for model in models['dna']['bpe']:
        logger.info(f' model [{model.name_or_path}] has {count_parameters(model):,} parameters')

    for model in models['dna']['kmer']:
        logger.info(f' model [{model.name_or_path}] has {count_parameters(model):,} parameters')


def plot_fisher(results):
    text = results['text']['bpe']['fisher']
    dna_bpe = results['dna']['bpe']['fisher']
    dna_kmer = results['dna']['kmer']['fisher']

    plot_fisher_information(text, dna_bpe, dna_kmer)

    text_full = results['text']['bpe']['fisher_full']
    dna_bpe_full = results['dna']['bpe']['fisher_full']
    dna_kmer_full = results['dna']['kmer']['fisher_full']

    plot_full_fisher_information(text_full, dna_bpe_full, dna_kmer_full)


def plot_distributions(results):
    text_js = results['text']['bpe']['js']
    dna_bpe_js = results['dna']['bpe']['js']
    dna_kmer_js = results['dna']['kmer']['js']
    plot_jensen_shannon(text_js, dna_bpe_js, dna_kmer_js)

    n_text = 10
    n_dna = 50

    text_mean_dist = results['text']['bpe']['mean_dist'][:n_text]
    dna_bpe_mean_dist = results['dna']['bpe']['mean_dist'][:n_dna]
    dna_kmer_mean_dist = results['dna']['kmer']['mean_dist'][:n_dna]
    plot_average_distribution(text_mean_dist, dna_bpe_mean_dist, dna_kmer_mean_dist)

if __name__ == '__main__': main()
