from torch.multiprocessing import cpu_count
from transformers import BertConfig, TrainingArguments 

def get_training_args(seed: int, **args):
    '''
    Wrapper for `transformers.TrainingArguments`.

    Applies args and ensures random model initialization but deterministic data.
    '''

    workers = min(cpu_count() // 2, args['batch_size'])

    return TrainingArguments(
        # batch size
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],

        # epochs
        num_train_epochs=args['epochs'],

        # keep model that performs best on unseen data
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True, # load the best model (to be saved) at the end
        greater_is_better=False,
        save_strategy='epoch', # save model at the end of each epoch

        # use many workers for dl
        dataloader_num_workers=workers,

        logging_strategy='epoch',
        eval_strategy='epoch',

        # lr
        learning_rate=5e-4,

        # warmup
        warmup_ratio=0.1,

        # wandb
        report_to='wandb', # log to wandb
        logging_steps=1,   # how often to log to wandb

        # seeds (random model init but not data)
        seed=seed,
        data_seed=42, # deterministic dataloader
    )

# base configuration for training on 1b tokens
base = {
    'N': 5,    # number of models to train
    'kmer': 6, # vocab size = 4**kmer + special tokens

    'epochs': 5,
    'batch_size': 256,
    'max_length': 512,
    'eval_size': 10_000,
    'train_size': 1_000_000,
    # 5 * 512 * 1M ~ 2.5b tokens
}

_test = { # very small config for testing
    'N': 5,
    'kmer': 6,
    'epochs': 2,
    'batch_size': 16,
    'max_length': 512,
    'eval_size': 50,
    'train_size': 200,
}

# base = _test; print('USING DUMMY TEST CONFIG') # uncomment for testing that everything works properly

# the 3 functions below only change the size of the BERT model used.

def get_config_90M() -> dict:
    '''
    Get config with a 90M params BERT model.

    (This one simply returns the default BERT configuration as provided by HuggingFace.)
    '''
    config = base.copy()
    config['bertconfig'] = BertConfig()
    return config

def get_config_20M() -> dict:
    '''Get config with a 20M params BERT model.'''
    config = base.copy()
    config['bertconfig'] = BertConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
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
    )
    return config
