import torch
import pprint
import argparse

from src.utils import get_logger, get_config_4M, get_config_20M, get_config_90M, get_config_90M_noT
from .train import train


def main() -> None:
    cmdargs = parse_cmdline_args()
    logger = get_logger('<train>')

    match cmdargs.size: # dispatch model config

        case '90M': 
            logger.info(' using 90M parameters config')
            args = get_config_90M()

        case '20M': 
            logger.info(' using 20M parameters config')
            args = get_config_20M()

        case '4M': 
            logger.info(' using 4M parameters config')
            args = get_config_4M()

    args['logger'] = logger
    args['tokenizer_name'] = cmdargs.tokenizer
    args['description'] = cmdargs.description
    args['data'] = cmdargs.data

    logger.info(f' training on {cmdargs.type}')
    logger.info(f' training on {args["epochs"]*args["train_size"]*args["max_length"]:,} tokens')
    logger.info(f' available gpus: {torch.cuda.device_count()}')
    logger.info(f' args:\n{pprint.pformat(args, indent=0, underscore_numbers=True)}')

    train(cmdargs.type, **args)


def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='Train N BERT models on either text or dna.')
    parser.add_argument('--type', type=str, required=True, 
                        choices=['text', 'dna'], help='whether to train on text or dna')
    parser.add_argument('--data', type=str, required=True,
                        choices=['wiki', 'og2', 'ncrna', 'cdna'], help='which dataset to train on')
    parser.add_argument('--tokenizer', type=str, required=True, 
                        choices=['kmer', 'bpe'], help='the tokenizer to use, bpe or k-mer')
    parser.add_argument('--size', type=str, required=False, default='90M', 
                        choices=['4M', '20M', '90M'], help='what bert config to use [small, medium, large]')
    parser.add_argument('--description', type=str, required=False, default=None,
                        help='optional description to add to the run path')
    return parser.parse_args()


if __name__ == '__main__': main()
