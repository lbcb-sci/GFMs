import argparse
from pathlib import Path
from pprint import pprint
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer

import src.analyze as analyze
from src.analyze.plotting import *
from src.utils import get_logger, count_parameters, DATA_TOKENIZER_PAIRS, get_plot_stem, run_key


def main() -> None:
    args = parse_args()

    logger = get_logger('<analyze>'); args.logger = logger
    stem = get_plot_stem(args.description)

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
            plot_fisher(results, stem)

        case 'distribution':
            logger.info(' running distributions analysis...')
            results = analyze.distributions(models, tokenizers, args)
            logger.info(' distributions analysis done\n')
            pprint(results)
            plot_distributions(results, stem)

        case 'attention':
            logger.info(' running attention scores analysis...')
            results = analyze.attention(models, tokenizers, args)
            logger.info(' attention scores analysis done\n')
            pprint(results)
            plot_attention(results, stem)

        case 'activations':
            logger.info(' running activations analysis...')
            results = analyze.activations(models, tokenizers, args)
            logger.info(' activations analysis done\n')
            pprint(results)
            print_results(results)

        case 'embeddings':
            logger.info(' running embeddings analysis...')
            results = analyze.embeddings(models, tokenizers, args)
            logger.info(' embeddings analysis done\n')
            pprint(results)
            print_results(results)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the trained models.')

    parser.add_argument('--type', type=str, required=True, choices=['static', 'distribution', 'fisher', 'attention', 'activations', 'embeddings'],
                        help='type of analysis to perform')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--description', type=str, required=False, default=None,
                        help='optional description to add to the run path')

    return parser.parse_args()


def get_huggingface_paths() -> dict:
    N = 5; username = 'mrochk'
    result = {}

    for data, tok, type in DATA_TOKENIZER_PAIRS:
        key = run_key(data, tok, type)
        result[key] = [f'{username}/bert-90M-{data}-{tok}-{idx}' for idx in range(1, N + 1)]

    return result


def get_local_paths(shuffle: bool) -> dict:
    N = 5
    base = Path('/home/vrcekl/scratch/GFMs/analyze/')
    suffix = 'shuffle' if shuffle else 'nonshuffle'
    result = {}

    for data, tok, type in DATA_TOKENIZER_PAIRS:
        key = run_key(data, tok, type)
        type_suffix = f'_{type}' if type else ''
        result[key] = [base / f'{data}_{tok}_{suffix}{type_suffix}' / str(idx) for idx in range(1, N + 1)]

    return result


def print_results(results: dict) -> None:
    for key, metrics in results.items():
        print(key)
        print('-' * 20)
        for metric, value in metrics.items():
            print(f'{metric}\t\t{value}')


def load_tokenizers(paths: dict) -> dict:
    result = {}
    for data, tok, type in DATA_TOKENIZER_PAIRS:
        key = run_key(data, tok, type)
        result[key] = [AutoTokenizer.from_pretrained(path, local_files_only=True) for path in paths[key]]
    return result


def load_models(paths: dict, device) -> dict:
    models = {}
    for data, tok, type in DATA_TOKENIZER_PAIRS:
        key = run_key(data, tok, type)
        models[key] = []
        for path in paths[key]:
            config = BertConfig.from_pretrained(path)
            config.output_attentions = True
            config.output_hidden_states = True
            model = BertForMaskedLM.from_pretrained(path, config=config, device_map='cpu').eval()
            models[key].append(model)
    return models


def check_models(models: dict, logger) -> None:
    for key, model_list in models.items():
        for model in model_list:
            logger.info(f' [{key}] model [{model.name_or_path}] has {count_parameters(model):,} parameters')


def plot_attention(results, stem: str = ''):
    text    = results['text_bpe_wiki']['mimatrix']
    dna_bpe = results['dna_bpe_OG2']['mimatrix']
    dna_kmer = results['dna_kmer_OG2']['mimatrix']
    plot_attention_scores(text, dna_bpe, dna_kmer, stem)

    text    = results['text_bpe_wiki']['entropies']
    dna_bpe = results['dna_bpe_OG2']['entropies']
    dna_kmer = results['dna_kmer_OG2']['entropies']
    plot_attention_entropies(text, dna_bpe, dna_kmer, stem)

    text     = results['text_bpe_wiki']['mimatrix_full']
    dna_bpe  = results['dna_bpe_OG2']['mimatrix_full']
    dna_kmer = results['dna_kmer_OG2']['mimatrix_full']
    plot_mi_matrix_full(text, dna_bpe, dna_kmer, 5, 12, stem=stem)

def plot_fisher(results, stem: str = ''):
    plot_fisher_information(results, stem)
    plot_full_fisher_information(results, stem)


def plot_distributions(results, stem: str = ''):
    plot_kl_divergence(results, stem)

    # text_js    = results['text_bpe']['js']
    # dna_bpe_js = results['dna_bpe_OG2']['js']
    # dna_kmer_js = results['dna_kmer_OG2']['js']
    # plot_jensen_shannon(text_js, dna_bpe_js, dna_kmer_js, stem)

    plot_average_distribution(results, stem)

if __name__ == '__main__': main()
