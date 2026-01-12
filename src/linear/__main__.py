import torch
import argparse

from src.models.linear import Linear 
from src.models.lstm import LSTM 
from src.tokenizer import KMERTokenizer
from src.datasets import nt_tasks, genomic_benchmarks
from src.common import (
    get_dataloader,
    get_logger,
    device,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['linear', 'lstm'])
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--tokens_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=False, default=20)
    return parser.parse_args()

@torch.no_grad()
def eval_model(dataloader, model, tokenizer):
    model = model.eval()
    all_preds = []; all_labels = []

    for _, (sequences, labels) in enumerate(dataloader):
        tokens = tokenizer(sequences).to(device)
        logits = model(tokens)
        preds  = torch.softmax(logits, dim=-1).argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0: mcc = 0.0
    else: mcc = ((tp * tn) - (fp * fn)) / denom  # [web:92][web:95]

    denom_f1 = (2 * tp + fp + fn)
    if denom_f1 == 0: f1 = 0.0
    else: f1 = 2 * tp / denom_f1

    model = model.train()
    return accuracy, mcc, f1

def main(task: str = None):
    torch.multiprocessing.set_sharing_strategy('file_system')

    logger = get_logger('embeddings')
    logger.info(f' using device {device}')

    # parse args
    args = get_args()
    logger.info(f' args = {args}')

    if task is None: task = args.task
    model = args.model
    epochs = args.epochs
    kmer = args.tokens_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    tokenizer = KMERTokenizer(kmer)
    logger.info(f' vocab size = {tokenizer.vocab_size}')

    if model == 'linear':
        model = Linear(
            vocab_size=tokenizer.vocab_size,
            num_labels=2,
        ).to(device)
    elif model == 'lstm':
        model = LSTM(
            vocab_size=tokenizer.vocab_size,
            num_labels=2,
        ).to(device)
    else: raise Exception()

    print(model)

    # get data
    logger.info(f' loading dataset {task}...')

    dataloader = get_dataloader(
        task=task, 
        split='train',
        batch_size=batch_size, 
        num_workers=num_workers,
    )

    dataloader_test = get_dataloader(
        task=task, 
        split='test',
        batch_size=batch_size, 
        num_workers=num_workers,
    )

    logger.info(' loading dataset done.')

    # compute embeddings
    logger.info(f' starting training...')

    lossfunc  = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, epochs+1):
        for _, (sequences, labels) in enumerate(dataloader):
            tokens = tokenizer(sequences).to(device) 
            logits = model(tokens)
            loss   = lossfunc(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc, mcc, f1 = eval_model(dataloader_test, model, tokenizer)
        logger.info(f' [{task}] epoch {epoch}/{epochs}: loss={loss.item():.2f}, mcc={mcc:.2f}, f1={f1:.2f}, acc={acc:.2f}')

if __name__ == '__main__': 
    main()
