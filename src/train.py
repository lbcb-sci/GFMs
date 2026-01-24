import argparse
import random
import numpy
import torch

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
    AutoTokenizer,
    BertConfig, 
    Trainer, 
    set_seed,
)

from src.instability.config import train_args, get_training_args
from src.instability.tokenizer import train_bpe_tokenizer, make_iterator, clean_text
from src.common import get_models_path, print_parameters

def train(bertconfig: BertConfig, tokenizer, collator, train_encoded, eval_encoded, prefix: str, **args):
    N = args['n_models']
    
    def mkpath(seed: int):
        tok       = args["tokenizer_name"]
        vsize     = bertconfig.vocab_size
        trainsize = args["train_size"]
        epochs    = args["epochs"]
        maxlen    = args["max_length"]
        nhidden   = bertconfig.num_hidden_layers
        hsize     = bertconfig.hidden_size
        intsize   = bertconfig.intermediate_size
        return get_models_path() / f'{prefix}_{seed}_{tok}_{vsize}_{trainsize}_{epochs}_{maxlen}_{nhidden}_{hsize}_{intsize}'

    for seed in range(N):
        training_args = get_training_args(seed, **args)

        # making sure model gets different initialization every time!
        set_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = BertForMaskedLM(bertconfig)
        print_parameters(model)

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=collator,
            eval_dataset=eval_encoded,
            train_dataset=train_encoded,
        )

        trainer.train()
        trainer.save_model(output_dir=mkpath(seed))

def get_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

def train_llms(**args):
    assert args['tokenizer_name'] == 'bpe'

    train_size = args['train_size']
    eval_size  = args['eval_size']

    trainset = f'train[:{train_size}]'
    evalset  = f'train[{train_size}:{train_size + eval_size}]'

    dataset_name = 'wikimedia/wikipedia'
    dataset_train = load_dataset(dataset_name, '20231101.en', split=trainset)
    dataset_eval  = load_dataset(dataset_name, '20231101.en', split=evalset)

    bertconfig: BertConfig = args['config']

    tokenizer = train_bpe_tokenizer(make_iterator(dataset_train), bertconfig.vocab_size)

    bertconfig.vocab_size   = tokenizer.vocab_size
    bertconfig.pad_token_id = tokenizer.pad_token_id
    bertconfig.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    bertconfig.eos_token_id = getattr(tokenizer, 'eos_token_id', None)

    print('first 20 tokens of vocab:', list(tokenizer.get_vocab().keys())[:20])

    def preprocess(batch):
        cleaned = [clean_text(t) for t in batch["text"]]
        return tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=args["max_length"],
        )

    remove = ['text', 'url', 'id', 'title']

    train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=remove)
    eval_encoded  = dataset_eval.map(preprocess,  batched=True, remove_columns=remove)
    data_collator = get_collator(tokenizer=tokenizer)

    return train(
        bertconfig, tokenizer, data_collator, 
        train_encoded, eval_encoded, 
        'llm', **args,
    )

def train_glms(**args):
    train_size = args['train_size']
    eval_size  = args['eval_size']

    trainset = f'train[:{train_size}]'
    evalset  = f'train[{train_size}:{train_size + eval_size}]'

    dataset_name  = 'zhangtaolab/plant-reference-genomes'
    dataset_train = load_dataset(dataset_name, split=trainset)
    dataset_eval  = load_dataset(dataset_name, split=evalset)

    bertconfig: BertConfig = args['config']

    match args['tokenizer_name']:

        case 'bpe':
            tokenizer = train_bpe_tokenizer(make_iterator(dataset_train), bertconfig.vocab_size)
        
        case 'ovl':
            assert bertconfig.vocab_size == 4**6
            tokenizer = AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-2.5b-multi-species')

        case _: raise Exception('tokenizer not supported')

    bertconfig.vocab_size   = tokenizer.vocab_size
    bertconfig.pad_token_id = tokenizer.pad_token_id
    bertconfig.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    bertconfig.eos_token_id = getattr(tokenizer, 'eos_token_id', None)

    print(tokenizer)
    print('first 20 tokens of vocab:', list(tokenizer.get_vocab().keys())[:20])

    preprocess = lambda batch: tokenizer(
        batch['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=args['max_length'],
    )

    train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=['text'])
    eval_encoded  = dataset_eval.map(preprocess,  batched=True, remove_columns=['text'])
    data_collator = get_collator(tokenizer=tokenizer)

    return train(
        bertconfig=bertconfig, tokenizer=tokenizer, collator=data_collator, 
        train_encoded=train_encoded, eval_encoded=eval_encoded, 
        prefix='glm', **args,
    )

def parse_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['llm', 'glm'])
    parser.add_argument('--n_models', type=int, required=False, default=5)
    parser.add_argument('--tokenizer', type=str, required=False, default='bpe', choices=['ovl', 'bpe'])
    parser.add_argument('--vocab_size', type=int, required=False, default=6)
    args = parser.parse_args()
    return args

def main():
    args = parse_cmdline_args(); print(args)
    train_args['n_models'] = args.n_models
    train_args['tokenizer_name']= args.tokenizer

    length = train_args['max_length']
    size = train_args['train_size']
    epochs = train_args['epochs']
    print(f'training on {size*length*epochs / 1_000_000:.1f}M tokens.')

    vocab_size = 4**args.vocab_size
    train_args['config'].vocab_size = vocab_size

    print(train_args)

    match args.type:

        case 'llm': 
            print('TRAINING LLMS')
            train_llms(**train_args)

        case 'glm': 
            print('TRAINING GLMS')
            train_glms(**train_args)

        case _ : exit(1)

if __name__ == '__main__': main()
