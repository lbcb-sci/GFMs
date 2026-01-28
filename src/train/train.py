import numpy
import torch
import random
from datasets import Dataset
from transformers import BertForMaskedLM, BertConfig, Trainer, set_seed
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .tokenizer import train_bpe_tokenizer, load_6mer_tokenizer, make_iterator, clean_text
from src.utils import count_parameters, get_run_path, get_training_args
from .data import get_dataset_text, get_dataset_dna

def train(type: str, **args) -> None:
    is_text = type == 'text'

    logger = args['logger']
    tokenizer_name = args['tokenizer_name']

    if is_text and tokenizer_name != 'bpe': 
        logger.fatal(' for text only bpe tokenizer is supported')
        exit(1)

    logger.info(f' collecting dataset...')

    train_size = args['train_size']
    eval_size  = args['eval_size']

    dataset_train, dataset_eval = \
        get_dataset_text(train_size, eval_size) if is_text \
        else get_dataset_dna(train_size, eval_size)

    logger.info(' collecting dataset done.')

    bertconfig = args['bertconfig']
    vocab_size = 4**args['kmer']

    logger.info(' getting tokenizer...')

    if tokenizer_name == 'bpe': 
        logger.info(' training bpe tokenizer...')
        tokenizer = train_bpe_tokenizer(make_iterator(dataset_train), vocab_size=vocab_size)
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

    print(tokenizer)

    logger.info(f' getting tokenizer done. vocab size = {tokenizer.vocab_size}')
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

    remove = ['text', 'url', 'id', 'title'] if is_text else ['text']

    train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=remove)
    eval_encoded  = dataset_eval.map(preprocess,  batched=True, remove_columns=remove)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    logger.info(' preprocessing dataset done.')
    logger.info(' calling main train function...')

    args.pop('bertconfig')

    return _train(bertconfig, tokenizer, data_collator, train_encoded, eval_encoded, type, **args)

def _train(
    bertconfig: BertConfig, 
    tokenizer: PreTrainedTokenizer, 
    collator: DataCollatorForLanguageModeling, 
    train_dataset: Dataset, 
    eval_dataset: Dataset, 
    prefix: str, 
    **args,
) -> None:

    '''Core train function.'''

    logger = args['logger']
    N = args['N']
    save_path = get_run_path(prefix, args['tokenizer_name'])

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
