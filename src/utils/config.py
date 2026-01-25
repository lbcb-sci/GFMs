from transformers import BertConfig, TrainingArguments 
from torch.multiprocessing import cpu_count

def get_training_args(seed: int, **args):
    '''Wrapper for transformers.TrainingArguments.'''

    workers = min(cpu_count() - 1, args['batch_size'])

    return TrainingArguments(
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        num_train_epochs=args['epochs'],
        metric_for_best_model='eval_loss',
        dataloader_num_workers=workers,
        load_best_model_at_end=True,
        greater_is_better=False,
        logging_strategy='epoch',
        eval_strategy='epoch',
        save_strategy='best',
        learning_rate=5e-4,
        warmup_ratio=0.1,
        seed=seed,
    )

# base configuration for training on 1b tokens
base = {
    'N': 5,    # number of models to train
    'kmer': 6, # vocab size = 4**kmer + special tokens

    'epochs': 10,
    'batch_size': 128,
    'max_length': 512,
    'eval_size': 5_000,
    'train_size': 200_000,
    # 10 * 512 * 200'000 = 1.024b tokens
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
    get config with a 90M params bert model
    this one simply returns the default bert configuration as provided by huggingface
    '''
    config = base.copy()
    config['bertconfig'] = BertConfig()
    return config

def get_config_20M() -> dict:
    '''get config with a 20M params bert model'''
    config = base.copy()
    config['bertconfig'] = BertConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
    )
    return config

def get_config_4M() -> dict:
    '''get config with a 4M params bert model'''
    config = base.copy()
    config['bertconfig'] = BertConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
    )
    return config
