import numpy
import torch
import random
import argparse
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    BertForMaskedLM, AutoTokenizer,
    BertConfig, Trainer, set_seed,
)

from src.tokenizer import train_bpe_tokenizer, make_iterator, clean_text
from src.utils import get_config_4M, get_config_20M, get_config_90M
from src.utils import (
    get_training_args, count_parameters,
    make_run_path, get_logger,
)

def train(
        bertconfig: BertConfig, 
        tokenizer, collator, 
        train_dataset, eval_dataset, 
        prefix: str, **args,
    ):

    logger = args['logger']
    N = args['N']

    save_path = make_run_path(prefix, args['tokenizer_name'])

    for seed in range(1, N+1):
        training_args = get_training_args(seed, **args)

        # making sure model gets different initialization every time!
        set_seed(seed); random.seed(seed); numpy.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        model = BertForMaskedLM(bertconfig)
        nparams = count_parameters(model)
        logger.info(f' model has {nparams:,} parameters')
        args['n_params'] = nparams

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=collator,
            eval_dataset=eval_dataset,
            train_dataset=train_dataset,
        )

        logger.info(f' starting the training of model #{seed}...')
        trainer.train()

        logger.info(f' model #{seed} trained')

        output = save_path / str(seed)
        trainer.save_model(output_dir=output)
        with open(output / 'configuration.txt', 'w') as c: c.write(str(args))
        logger.info(f' saved model at {output}')

def get_collator(tokenizer):
    '''define the MLM config here if needed'''
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

def train_text(**args):
    logger = args['logger']

    if args['tokenizer_name'] != 'bpe': 
        logger.fatal('for text only bpe tokenizer is supported')
        exit(1)

    dataset_name = 'wikimedia/wikipedia'
    logger.info(f' collecting dataset {dataset_name}...')

    train_size = args['train_size']
    eval_size  = args['eval_size']

    trainset = f'train[:{train_size}]'
    evalset  = f'train[{train_size}:{train_size + eval_size}]'

    dataset_train = load_dataset(dataset_name, '20231101.en', split=trainset)
    dataset_eval  = load_dataset(dataset_name, '20231101.en', split=evalset)

    logger.info(' collecting dataset done.')

    bertconfig = args['bertconfig']

    logger.info(' training bpe tokenizer...')

    vocab_size = 4**args['kmer']

    tokenizer = train_bpe_tokenizer(make_iterator(dataset_train), vocab_size=vocab_size)
    bertconfig.vocab_size   = tokenizer.vocab_size
    bertconfig.pad_token_id = tokenizer.pad_token_id
    bertconfig.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    bertconfig.eos_token_id = getattr(tokenizer, 'eos_token_id', None)

    logger.info(' training bpe tokenizer done.')
    logger.info(f' first 20 tokens: {list(tokenizer.get_vocab().keys())[:20]}')

    def preprocess(batch):
        cleaned = [clean_text(t) for t in batch["text"]]
        return tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=args["max_length"],
        )

    logger.info(' preprocessing dataset with tokenizer...')

    remove = ['text', 'url', 'id', 'title']

    train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=remove)
    eval_encoded  = dataset_eval.map(preprocess,  batched=True, remove_columns=remove)
    data_collator = get_collator(tokenizer=tokenizer)

    logger.info(' preprocessing dataset with tokenizer done.')

    logger.info(' calling train function')

    args.pop('bertconfig')

    return train(
        bertconfig, tokenizer, data_collator, 
        train_encoded, eval_encoded, 
        'text', **args,
    )

def train_dna(**args):
    logger = args['logger']

    dataset_name  = 'zhangtaolab/plant-reference-genomes'
    logger.info(f' collecting dataset {dataset_name}...')

    train_size = args['train_size']
    eval_size  = args['eval_size']

    trainset = f'train[:{train_size}]'
    evalset  = f'train[{train_size}:{train_size + eval_size}]'

    dataset_train = load_dataset(dataset_name, split=trainset)
    dataset_eval  = load_dataset(dataset_name, split=evalset)

    logger.info(' collecting dataset done.')

    bertconfig = args['bertconfig']

    match args['tokenizer_name']:

        case 'bpe':
            logger.info(' training bpe tokenizer...')
            vocab_size = 4**args['kmer']
            tokenizer = train_bpe_tokenizer(make_iterator(dataset_train), vocab_size=vocab_size)
            logger.info(' training bpe tokenizer done.')
        
        case 'ovl':
            if args['kmer'] != 6: 
                logger.fatal(' training on dna with ovl tokenizer is only implemented for 6-mers')
                exit(1)

            logger.info(f' loading overlapping {args["kmer"]}-mer tokenizer...')
            tokenizer = AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-2.5b-multi-species')
            logger.info(f' loading tokenizer done.')

        case _: 
            logger.fatal(' tokenizer not supported')
            exit(1)

    bertconfig.vocab_size   = tokenizer.vocab_size
    bertconfig.pad_token_id = tokenizer.pad_token_id
    bertconfig.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    bertconfig.eos_token_id = getattr(tokenizer, 'eos_token_id', None)

    logger.info(f' first 20 tokens: {list(tokenizer.get_vocab().keys())[:20]}')

    preprocess = lambda batch: tokenizer(
        batch['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=args['max_length'],
    )

    logger.info(' preprocessing dataset with tokenizer...')

    train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=['text'])
    eval_encoded  = dataset_eval.map(preprocess,  batched=True, remove_columns=['text'])
    data_collator = get_collator(tokenizer=tokenizer)

    logger.info(' preprocessing dataset with tokenizer done.')

    logger.info(' calling train function')

    args.pop('bertconfig')

    return train(
        bertconfig=bertconfig, tokenizer=tokenizer, collator=data_collator, 
        train_dataset=train_encoded, eval_dataset=eval_encoded, 
        prefix='dna', **args,
    )

def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='Train N BERT models on either text or dna.')
    parser.add_argument('--type', type=str, required=True, choices=['text', 'dna'], help='whether to train on text or dna')
    parser.add_argument('--tokenizer', type=str, required=True, choices=['ovl', 'bpe'], help='which tokenizer to use, bpe or overlapping k-mer')
    args = parser.parse_args()
    return args

def main():
    args = get_config_4M()

    cmdargs = parse_cmdline_args()
    args['tokenizer_name'] = cmdargs.tokenizer

    logger = get_logger('train')
    args['logger'] = logger

    for k, v in args.items(): logger.info(f' {k}={v}')

    match cmdargs.type: # dispatch to correct data modality

        case 'text': 
            logger.info(' training on text')
            train_text(**args)

        case 'dna': 
            logger.info(' training on dna')
            train_dna(**args)

        case _ : 
            logger.fatal(' data modality not supported')
            exit(1)

if __name__ == '__main__': main()
