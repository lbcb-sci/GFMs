import argparse
import numpy
import random
import torch
import re
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
    BertConfig, 
    Trainer, 
    set_seed,
)

from src.instability.tokenizer import train_bpe_tokenizer
from src.instability.config import train_args, get_training_args
from src.common import get_models_path

ALLOWED = r"[^a-zA-Z0-9\s.,;:!?\"'()\-–—/\\&%$€@#\[\]{}<>]+"

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(ALLOWED, " ", s)
    return s.strip()

def make_iterator(dataset):
    def iterator():
        for example in dataset:
            text = example["text"]
            if not isinstance(text, str): continue
            text = clean_text(text)
            if text.strip(): yield text
    return iterator

def train(bertconfig, tokenizer, collator, train_encoded, eval_encoded, prefix: str, **args):
    N = args['n_models']
    models = []

    path = get_models_path()

    for seed in range(N):
        training_args = get_training_args(seed, **args)

        # making sure model gets different initialization every time!
        set_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = BertForMaskedLM(bertconfig)

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=collator,
            eval_dataset=eval_encoded,
            train_dataset=train_encoded,
        )

        trainer.train()
        trainer.save_model(output_dir=path / f'{prefix}_{seed}')

        models.append(model)

    return models

def get_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

def train_llms(**args):

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

    print(list(tokenizer.get_vocab().keys())[:20])

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
    eval_size = args['eval_size']

    trainset = f'train[:{train_size}]'
    evalset = f'train[{train_size}:{train_size + eval_size}]'

    dataset_name = 'zhangtaolab/plant-reference-genomes'
    dataset_train = load_dataset(dataset_name, split=trainset)
    dataset_eval  = load_dataset(dataset_name, split=evalset)

    bertconfig: BertConfig = args['config']

    tokenizer = train_bpe_tokenizer(make_iterator(dataset_train), bertconfig.vocab_size)

    print(list(tokenizer.get_vocab().keys())[:10])

    preprocess = lambda batch: tokenizer(
        batch['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=args['max_length'],
    )

    train_encoded = dataset_train.map(preprocess, batched=True)
    eval_encoded  = dataset_eval.map(preprocess,  batched=True)
    data_collator = get_collator(tokenizer=tokenizer)

    return train(
        bertconfig, tokenizer, data_collator, 
        train_encoded, eval_encoded, 
        'glm', **args,
    )

def parse_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['llm', 'glm'])
    parser.add_argument('--n_models', type=int, required=False, default=5)
    args = parser.parse_args()
    return args

def main():
    args = parse_cmdline_args(); print(args)
    train_args['n_models'] = args.n_models

    length = train_args['max_length']
    size = train_args['train_size']
    epochs = train_args['epochs']
    print(f'training on {size*length*epochs / 1_000_000:.1f}M tokens.')

    match args.type:
        case 'llm': train_llms(**train_args)
        case 'glm': train_glms(**train_args)
        case     _: exit(1)

if __name__ == '__main__': main()
