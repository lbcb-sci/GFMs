import torch
from transformers import BertConfig, TrainingArguments 

train_args = {
    'epochs': 10,
    'batch_size': 128,
    'max_length': 512,   # max sequence length
    'train_size': 200_000, # number of sequences in training set
    'eval_size': 5_000,  # number of sequences in val set
    'config': BertConfig(
        #hidden_size=512,
        #intermediate_size=1024, # uncomment to make model smaller, otherwise
        #num_hidden_layers=8,    # use default BERT config, ~90M params
        #num_attention_heads=8,
    ),
}

'''
To calculate the number of training tokens:
tokens = epochs*train_size*max_length  (should be roughly 1b tokens)
'''

def get_training_args(seed: int, **args):
    workers = min(torch.multiprocessing.cpu_count()-1, args['batch_size'])
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
