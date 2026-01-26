import numpy
import torch
import random
import argparse
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, AutoTokenizer
from transformers import BertForMaskedLM, BertConfig, Trainer, set_seed

from src.tokenizer import train_bpe_tokenizer, make_iterator, clean_text
from src.utils import count_parameters, make_run_path, get_logger
from src.utils import get_training_args, get_config_4M, get_config_20M, get_config_90M

def get_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorForLanguageModeling:
    '''Make MaskedLM data collator.'''
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

def train(
    bertconfig: BertConfig, 
    tokenizer: PreTrainedTokenizer, 
    collator: DataCollatorForLanguageModeling, 
    train_dataset: Dataset, eval_dataset: Dataset, 
    prefix: str, **args,
) -> None:

    '''Core train function.'''

    logger = args['logger']
    N = args['N']

    save_path = make_run_path(prefix, args['tokenizer_name'])

    for seed in range(1, N+1):
        training_args = get_training_args(seed, **args)

        # making sure model gets different initialization every time!
        set_seed(seed); random.seed(seed); numpy.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        model = BertForMaskedLM(bertconfig).to(args['device'])
        logger.info(f' model device: {next(model.parameters()).device}')

        nparams = count_parameters(model)
        args['n_params'] = nparams
        logger.info(f' model has {nparams:,} parameters')

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

def train_text(**args):
    logger = args['logger']

    if args['tokenizer_name'] != 'bpe': 
        logger.fatal(' for text only bpe tokenizer is supported')
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
    parser.add_argument('--size', type=str, required=True, choices=['4M', '20M', '90M'], help='what bert config to use [small, medium, large]')
    parser.add_argument('--gpu', type=int, required=True, help='which gpu id to use for training')
    return parser.parse_args()

def main() -> None:

    import warnings, pprint

    warnings.simplefilter('ignore')

    cmdargs = parse_cmdline_args()

    logger = get_logger('<train>')

    match cmdargs.size:

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

    gpu = cmdargs.gpu
    device = f'cuda:{gpu}'
    args['device'] = device
    torch.cuda.set_device(gpu)

    logger.info(f' available gpus: {torch.cuda.device_count()}')
    logger.info(f' current cuda device: {torch.cuda.current_device()} ({device})')

    logger.info(f' args:\n{pprint.pformat(args, indent=0, underscore_numbers=True)}')

    match cmdargs.type: # dispatch to correct data modality

        case 'text': 
            logger.info(' training on text')
            train_text(**args)

        case 'dna': 
            logger.info(' training on dna')
            train_dna(**args)

if __name__ == '__main__': main()
