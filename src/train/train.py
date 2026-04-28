import wandb
import numpy
import torch
import random
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import BertForMaskedLM, BertConfig, PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed
# from transformers.trainer_utils import get_last_checkpoint

from datasets import load_from_disk

# from src.train.tokenizer.tokenizer import clean_dna
from .tokenizer import train_bpe_tokenizer, load_6mer_tokenizer, make_iterator_text, make_iterator_dna, clean_text, clean_dna
from src.utils import count_parameters, get_run_path, get_training_args
from src.utils.paths import PATHS
from .data import get_dataset_text
from .callback import FisherCallback, ThroughputCallback


def load_saved_dataset(path: str | Path, train_size: int, eval_size: int, logger):
    dataset_full = load_from_disk(path)
    dataset_size = len(dataset_full)
    required_size = train_size + eval_size

    if dataset_size == 0:
        logger.fatal(f' dataset at {path} is empty')
        exit(1)

    if dataset_size == 1:
        logger.fatal(f' dataset at {path} has only one row, cannot create both train and eval splits')
        exit(1)

    actual_eval_size = min(eval_size, dataset_size - 1)
    actual_train_size = min(train_size, dataset_size - actual_eval_size)

    if dataset_size < required_size:
        logger.warning(
            f' dataset at {path} has {dataset_size:,} rows, below the requested {required_size:,}; '
            f'using {actual_train_size:,} rows for training and {actual_eval_size:,} for evaluation instead'
        )

    dataset_train = dataset_full.select(range(actual_train_size))
    dataset_eval = dataset_full.select(range(actual_train_size, actual_train_size + actual_eval_size))
    return dataset_train, dataset_eval


def train(type: str, **args) -> None:
    is_text = type == 'text'

    logger = args['logger']
    tokenizer_name = args['tokenizer_name']

    if is_text and tokenizer_name != 'bpe': 
        logger.fatal(' for text only bpe tokenizer is supported')
        exit(1)

    train_size = args['train_size']
    eval_size  = args['eval_size']

    bertconfig = args['bertconfig']
    vocab_size = 4**args['kmer']

    logger.info(' getting tokenizer...')

    if is_text:
        preprocessed_path = PATHS['wiki_dataset']
        logger.info(f' loading preprocessed text dataset from {preprocessed_path}')

        dataset_train, dataset_eval = load_saved_dataset(preprocessed_path, train_size, eval_size, logger)


        # dataset_train, dataset_eval = get_dataset_text(3*train_size, 3*eval_size)
        # logger.info(f' loaded text dataset with {len(dataset_train):,} training and {len(dataset_eval):,} evaluation examples')

        # dataset_train = dataset_train.map(lambda x: {'length': len(x['text'])})
        # dataset_train = dataset_train.sort('length', reverse=True).select(range(train_size))
        
        # dataset_eval = dataset_eval.map(lambda x: {'length': len(x['text'])})
        # dataset_eval = dataset_eval.sort('length', reverse=True).select(range(eval_size))
        
        # logger.info(f' filtered text dataset to {len(dataset_train):,} training and {len(dataset_eval):,} evaluation examples with longest texts')
        # logger.info(f' shortest training example has {min(dataset_train["length"])} characters, longest has {max(dataset_train["length"])} characters')
        # logger.info(f' shortest evaluation example has {min(dataset_eval["length"])} characters, longest has {max(dataset_eval["length"])} characters')

        # dataset_train = dataset_train.remove_columns('length')
        # dataset_eval = dataset_eval.remove_columns('length')

        text_bpe_path = PATHS['text_bpe_tokenizer']  # Path('/home/vrcekl/scratch/GFMs/bpe_text_tokenizer')
        if text_bpe_path.exists():
            logger.info(f' loading text bpe tokenizer from {text_bpe_path}')
            tokenizer = PreTrainedTokenizerFast.from_pretrained(text_bpe_path)
        else:
            logger.info(' training text bpe tokenizer...')
            tokenizer = train_bpe_tokenizer(make_iterator_text(dataset_train), vocab_size=vocab_size)
            tokenizer.save_pretrained(text_bpe_path)
            logger.info(f' saved text bpe tokenizer to {text_bpe_path}')

        tokenizer.model_max_length = args['max_length']
        bertconfig.vocab_size      = tokenizer.vocab_size
        bertconfig.pad_token_id    = tokenizer.pad_token_id
        bertconfig.bos_token_id    = getattr(tokenizer, 'bos_token_id', None)
        bertconfig.eos_token_id    = getattr(tokenizer, 'eos_token_id', None)

        logger.info(f' getting tokenizer done, vocab size = {tokenizer.vocab_size}')
        logger.info(f' first 20 tokens: {list(tokenizer.get_vocab().keys())[:20]}')
        logger.info(f' preprocessing dataset with tokenizer...')

        def preprocess(batch):
            cleaned = [clean_text(t) for t in batch["text"]]
            return tokenizer(cleaned, truncation=True, max_length=args["max_length"], padding=False)

        remove = ['text']
        if 'url' in dataset_train.column_names: remove.append('url')
        if 'id' in dataset_train.column_names: remove.append('id')
        if 'title' in dataset_train.column_names: remove.append('title')

        train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=remove)
        eval_encoded  = dataset_eval.map( preprocess, batched=True, remove_columns=remove)

        logger.info(f' shortest tokenized training example has {min([len(s) for s in train_encoded["input_ids"]])} tokens')
        logger.info(f' shortest tokenized evaluation example has {min([len(s) for s in eval_encoded["input_ids"]])} tokens\n')

    else:
        # preprocessed_path = PATHS['og2_dataset']
        preprocessed_path = Path(args['dna_dataset_path']) if args.get('dna_dataset_path') else PATHS['ensembl_cdna_chunks']
        logger.info(f' loading preprocessed DNA dataset from {preprocessed_path}')

        dataset_train, dataset_eval = load_saved_dataset(preprocessed_path, train_size, eval_size, logger)

        if tokenizer_name == 'bpe':
            dna_bpe_path = PATHS['dna_bpe_tokenizer']  # Path('/home/vrcekl/scratch/GFMs/bpe_dna_tokenizer')
            if dna_bpe_path.exists():
                logger.info(f' loading DNA bpe tokenizer from {dna_bpe_path}')
                tokenizer = PreTrainedTokenizerFast.from_pretrained(dna_bpe_path)
            else:
                logger.info(' training DNA bpe tokenizer...')
                tokenizer = train_bpe_tokenizer(make_iterator_dna(dataset_train), vocab_size=vocab_size)
                tokenizer.save_pretrained(dna_bpe_path)
                logger.info(f' saved DNA bpe tokenizer to {dna_bpe_path}')

        else:
            kmer = args['kmer']
            if kmer != 6:
                logger.fatal(' training on dna with kmer tokenizer is only implemented for 6-mers')
                exit(1)

            logger.info(f' loading {kmer}-mer tokenizer...')
            tokenizer = load_6mer_tokenizer()

        tokenizer.model_max_length = args['max_length']
        bertconfig.vocab_size      = tokenizer.vocab_size
        bertconfig.pad_token_id    = tokenizer.pad_token_id
        bertconfig.bos_token_id    = getattr(tokenizer, 'bos_token_id', None)
        bertconfig.eos_token_id    = getattr(tokenizer, 'eos_token_id', None)

        logger.info(f' getting tokenizer done, vocab size = {tokenizer.vocab_size}')
        logger.info(f' first 20 tokens: {list(tokenizer.get_vocab().keys())[:20]}')
        logger.info(f' preprocessing dataset with tokenizer...')

        def preprocess(batch):
            cleaned = [clean_dna(t) for t in batch["text"]]
            return tokenizer(cleaned, truncation=True, max_length=args["max_length"], padding=False)

        remove = ['text']
        train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=remove)
        eval_encoded  = dataset_eval.map(preprocess, batched=True, remove_columns=remove)

    logger.info(' preprocessing dataset done.')
    logger.info(' calling main train function...')

    args.pop('bertconfig')

    return _train(bertconfig, tokenizer, train_encoded, eval_encoded, type, **args)


def _train(
    bertconfig: BertConfig, 
    tokenizer, 
    train_dataset: Dataset, 
    eval_dataset: Dataset, 
    prefix: str, 
    **args,
) -> None:

    '''Core train function.'''

    logger = args['logger']
    logger.info(f' model config: {bertconfig}')
    N = args['N']
    args = args.copy()
    explicit_seed = args.pop('seed', None)
    run_description = args.get('description')
    if explicit_seed is not None:
        run_description = f'{run_description}_seed{explicit_seed}' if run_description else f'seed{explicit_seed}'

    save_path = get_run_path(prefix, args['tokenizer_name'], description=run_description)

    # last_ckpt = get_last_("")
    # print(f'Resumingf from:', last_ckpt)

    seeds = [explicit_seed] if explicit_seed is not None else range(1, N + 1)

    for seed in seeds:  # train N models with different seeds, or a single requested seed
        # making sure model gets different initialization every time!
        set_seed(seed); random.seed(seed); numpy.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        time_start = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
        logger.info(f' training of model with seed {seed} started at {time_start}')

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=0.15,
        )

        training_args = get_training_args(
            run_name=f'{str(save_path).split('/')[-1]}/{seed}', 
            seed=seed, 
            **args,
        )

        model = BertForMaskedLM(bertconfig)

        nparams = count_parameters(model)
        args['n_params'] = nparams
        logger.info(f' model has {nparams:,} parameters')

        keys = ['bert.embeddings', 'bert.encoder', 'cls']

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[FisherCallback(keys=keys, batch_size=args['batch_size']), ThroughputCallback(max_length=args['max_length'])]
        )

        output_init = save_path / str(seed) / 'init'
        output_trained = save_path / str(seed) / 'trained'

        trainer.save_model(output_dir=output_init)
        with open(output_init / 'configuration.txt', 'w') as c: c.write(str(args))
        logger.info(f' saved initialized model at {output_init}')

        logger.info(f' starting the training of model #{seed}...')
        trainer.train()

        logger.info(f' model #{seed} trained')

        trainer.save_model(output_dir=output_trained)
        with open(output_trained / 'configuration.txt', 'w') as c: c.write(str(args))
        logger.info(f' saved trained model at {output_trained}')

        time_end = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
        logger.info(f' training of model with seed {seed} started at {time_start} and ended at {time_end}')

        wandb.finish()
