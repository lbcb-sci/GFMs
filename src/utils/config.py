from torch.multiprocessing import cpu_count
from transformers import BertConfig, TrainingArguments

from src.utils.paths import PATHS


def get_training_args(run_name: str, seed: int, **args):
    '''
    Wrapper for `transformers.TrainingArguments`.

    Applies args and ensures random model initialization but deterministic data.
    '''

    workers = min(16, args['batch_size'])

    return TrainingArguments(
        # checkpoint location
        output_dir= PATHS['checkpoints'] / run_name,

        # batch size
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],

        use_cpu=False,

        # epochs
        num_train_epochs=args['epochs'],

        # keep model that performs best on unseen data
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True, # load the best model (to be saved) at the end
        greater_is_better=False,

        # use many workers for dl
        dataloader_num_workers=workers,

        save_strategy='steps',
        eval_strategy='steps',
        save_steps=10_000,
        eval_steps=1_000,

        fp16=False,
        bf16=False,

        label_smoothing_factor=0.0,

        # lr
        learning_rate=1e-4,

        warmup_ratio=0.1,
        lr_scheduler_type='cosine',

        # wandb
        report_to='wandb', # log to wandb
        run_name=run_name,
        logging_strategy='steps',
        logging_steps=100, # how often to log to wandb

        max_grad_norm=0.5,

        eval_on_start=False,

        # seeds (random model init but not data)
        seed=seed,
        data_seed=42, # non-deterministic dataloader, put 42 for deterministic
    )

N = 5

# base configuration for training on 1b tokens
base = {
    'N': N,    # number of models to train
    'kmer': 6, # vocab size = 4**kmer + special tokens
    'epochs': 10,
    'batch_size': 96,
    'max_length': 512,
    'eval_size': 10_000,
    'train_size': 1_000_000,
    # 10 * 512 * 2M ~ 10b target tokens; smaller datasets now fall back to their available size.
}

_test = { # very small config for testing
    'N': N,
    'kmer': 6,
    'epochs': 2,
    'batch_size': 16,
    'max_length': 512,
    'eval_size': 50,
    'train_size': 10000,
}

#base = _test; print('USING DUMMY TEST CONFIG') # uncomment for testing that everything works properly

# the 3 functions below only change the size of the BERT model used.

def get_config_90M() -> dict:
    '''
    Get config with a 90M params BERT model.

    (This one simply returns the default BERT configuration as provided by HuggingFace.)
    '''
    config = base.copy()
    config['bertconfig'] = BertConfig(vocab_size=0) # setting vocab size to 0 because it will be updated by tokenizer
    return config


def get_config_90M_noT() -> dict:
    '''Get config with a 90M params BERT model, but without the Transformer layers.'''
    config = base.copy()
    config['bertconfig'] = BertConfig(
        num_hidden_layers=0,
        vocab_size=0
    )
    return config


def get_config_20M() -> dict:
    '''Get config with a 20M params BERT model.'''
    config = base.copy()
    config['bertconfig'] = BertConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        vocab_size=0,
    )
    return config


def get_config_4M() -> dict:
    '''Get config with a 4M params BERT model.'''
    config = base.copy()
    config['bertconfig'] = BertConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        vocab_size=0,
    )
    return config
