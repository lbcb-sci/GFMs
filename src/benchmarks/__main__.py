import argparse
import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from src.common import get_dl, get_logger, device, get_results_path
from src.datasets import genomic_benchmarks, nt_tasks
from src.models import linear, markov
from src.tokenizer import KmerTokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True, choices=['gb', 'nt'])
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--kmer_from', type=int, required=False, default=1)
    parser.add_argument('--kmer_to', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--num_workers_dl', type=int, required=False, default=0)
    return parser.parse_args()

def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    acc = (tp + tn) / (tp + tn + fp + fn)

    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0: mcc = 0.0
    else: mcc = ((tp * tn) - (fp * fn)) / denom  # [web:92][web:95]

    denom_f1 = (2 * tp + fp + fn)
    if denom_f1 == 0: f1 = 0.0
    else: f1 = 2 * tp / denom_f1

    return {'acc': float(acc), 'mcc': float(mcc), 'f1': float(f1)}

def train_linear(dl: DataLoader, kmer: int, epochs: int) -> linear.LinearEmbedding:
    tokenizer = KmerTokenizer(kmer)
    model = linear.LinearCount(vocab_size=tokenizer.vocab_size, num_labels=2).to(device)
    lossfunc  = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, epochs+1):
        for _, (sequences, labels) in enumerate(dl):
            tokens = tokenizer(sequences).to(device) 
            logits = model(tokens)
            loss   = lossfunc(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.eval(), tokenizer

@torch.no_grad()
def eval_linear(dl: DataLoader, tokenizer: KmerTokenizer, model: linear.LinearEmbedding) -> dict:
    model = model.eval()
    all_preds = []; all_labels = []

    for _, (sequences, labels) in enumerate(dl):
        tokens = tokenizer(sequences).to(device)
        logits = model(tokens)
        preds  = torch.softmax(logits, dim=-1).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    return compute_metrics(tp, fp, tn, fn)

def train_markov(dl: DataLoader, kmer: int) -> tuple[markov.MarkovChain]:
    positives = []; negatives = []

    for (sequences, labels) in dl:
        for (sequence, label) in zip(sequences, labels):
            if label.item() == 0: negatives.append(sequence)
            elif label.item() == 1: positives.append(sequence)

    model_pos = markov.MarkovChain(kmer)
    model_neg = markov.MarkovChain(kmer)

    model_pos.fit(positives, smoothing=True)
    model_neg.fit(negatives, smoothing=True)

    return model_pos, model_neg

def eval_markov(dl: DataLoader, models: tuple[markov.MarkovChain]) -> dict:
    pos, neg = models
    tp = fp = tn = fn = 0

    for (sequences, labels) in dl:
        for (sequence, label) in zip(sequences, labels):
            label = label.item()

            llpos = pos.ll(sequence)
            llneg = neg.ll(sequence)

            prediction = llpos > llneg

            tp += prediction == label and label == 1
            fp += prediction != label and label == 1
            tn += prediction == label and label == 0
            fn += prediction != label and label == 0

    return compute_metrics(tp, fp, tn, fn)

def run_kmer(dl: DataLoader, dl_test: DataLoader, kmer: int, epochs: int) -> dict:
    linear, tokenizer = train_linear(dl, kmer, epochs)
    result_linear = eval_linear(dl_test, tokenizer, linear)

    markov = train_markov(dl, kmer)
    result_markov = eval_markov(dl_test, markov)
    
    return {'kmer': kmer, 'linear': result_linear, 'markov': result_markov}

def main():
    logger = get_logger('benchmarks')
    logger.info(f' using device {device}')

    args = get_args()
    logger.info(f' args = {args}')
    benchmark = args.benchmark
    epochs = args.epochs
    kmer_from = args.kmer_from
    kmer_to = args.kmer_to
    batch_size = args.batch_size
    num_workers_dl = args.num_workers_dl

    path = get_results_path()

    tasks = nt_tasks.BINARY_TASKS if benchmark == 'nt' else genomic_benchmarks.BINARY_TASKS
    for task in tasks:

        logger.info(f' loading trainset {task}...')
        dl = get_dl(task=task, split='train', batch_size=batch_size, num_workers=num_workers_dl)

        logger.info(f' loading testset {task}...')
        dl_test = get_dl(task=task, split='test', batch_size=batch_size, num_workers=num_workers_dl)

        logger.info(' running train and eval...')

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=mp.cpu_count()) as pool:
            args = [(dl, dl_test, kmer, epochs) for kmer in range(kmer_from, kmer_to+1)]
            results = pool.starmap(run_kmer, args)

        best_linear = max(results, key=lambda r: r["linear"]["mcc"])
        best_markov = max(results, key=lambda r: r["markov"]["mcc"])

        best_kmer_linear = best_linear["kmer"]
        best_kmer_markov = best_markov["kmer"]

        result_linear = best_linear["linear"]
        result_markov = best_markov["markov"]

        with open(path / f'linear.{benchmark}', 'a') as f:
            f.write(f'\n[{task}] ({best_kmer_linear}) {result_linear}')

        with open(path / f'markov.{benchmark}', 'a') as f:
            f.write(f'\n[{task}] ({best_kmer_markov}) {result_markov}')

        logger.info(f' {task} done.')

if __name__ == '__main__': 
    mp.set_start_method("spawn", force=True)
    main()
