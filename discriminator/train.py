import wandb
from pprint import pp
from tqdm import tqdm
from datasets import Dataset

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count
from torch.optim.lr_scheduler import CosineAnnealingLR

from discriminator.config import *
from discriminator.data import load_generated_dataset
from discriminator.model import Discriminator
from discriminator.utils import count_parameters
from discriminator.tokenizer import get_tokenizer

@torch.inference_mode()
def evaluate(model: Discriminator, testloader: DataLoader) -> dict:
    '''
    Computes classic binary classification metrics on given test dataloader.
    
    Returns (loss, accuracy, f1, precision, recall) in a dictionary.
    '''
    model = model.eval()

    tp = tn = fp = fn = sumloss = nsamples = 0

    for batch in tqdm(testloader, desc=f'Evaluating Discriminator'):
        labels = batch['real'].float().cuda()
        sequences = batch['input_ids'].cuda()

        logits = model(sequences)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
        sumloss += loss

        predictions = torch.sigmoid(logits)

        positive_preds  = (predictions > 0.5).int()
        negative_preds  = (predictions < 0.5).int()
        positive_labels = (labels > 0.5).int()
        negative_labels = (labels < 0.5).int()
        
        tp += (positive_preds & positive_labels).sum()
        tn += (negative_preds & negative_labels).sum()
        fp += (positive_preds & negative_labels).sum()
        fn += (negative_preds & positive_labels).sum()
        nsamples += labels.shape[0]

    accuracy  = ((tp + tn) / nsamples).item()
    precision = (tp / (tp + fp)).item()
    recall    = (tp / (tp + fn)).item()
    f1        = (2 * precision * recall) / (precision + recall)
    loss      = (sumloss / nsamples).item()

    model = model.train()
    return {'test/loss': loss, 'test/acc:': accuracy, 'test/precision': precision, 'test/recall': recall, 'test/f1': f1}

def train_on_split(
    model: Discriminator, 
    split: Dataset,
    optim: torch.optim.Optimizer, 
    ratio: float,
    run: wandb.Run, 
) -> Discriminator:
    '''
    Train the discriminator on sequences with the given randomized ratio (a number in [0, 1]).
    '''
    
    split = split.train_test_split(test_size=Ptrain.test_set_size)
    train, test = split['train'], split['test']

    print(train, test)

    trainloader = DataLoader(train, batch_size=Ptrain.batch_size, shuffle=True, num_workers=cpu_count() // 2)
    testloader  = DataLoader(test,  batch_size=Ptrain.batch_size, shuffle=True, num_workers=cpu_count() // 2)

    epochs = Ptrain.epochs[ratio]
    base_lr = Ptrain.base_lr[ratio]

    # reset optimizer and lr scheduler before each split
    for p in optim.param_groups: p['lr'] = base_lr
    scheduler = CosineAnnealingLR(optim, T_max=len(trainloader)*epochs)

    for epoch in range(epochs):

        bar = tqdm(trainloader, desc=f'Training Discriminator (ratio={ratio}, epoch={epoch+1}/{epochs})')

        for step, batch in enumerate(bar):
            if (step == 0) or ((step+1) % Ptrain.eval_steps) == 0:
                results = evaluate(model, testloader)
                run.log(results)
                bar.write(f'Step {step}: {str(results)}')

            labels = batch['real'].float().cuda()
            sequences = batch['input_ids'].cuda()

            logits = model(sequences)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # step
            optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Ptrain.max_grad_norm)
            optim.step()
            scheduler.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                acc = torch.mean(((probs > 0.5).int() == labels.int()).float()).item()
                lr = scheduler.get_last_lr()[0]
                run.log({
                    'train/loss': loss.item(),
                    'train/acc': acc,
                    'train/lr': lr,
                    'train/grad_norm': grad_norm.item(),
                })
                bar.set_postfix_str(f'loss={loss.item():.2f}, acc={acc*100:.0f}%, lr={lr:.6f}, grad_norm: {grad_norm.item():.2f}')

        results = evaluate(model, testloader)
        run.log(results)
        bar.write(f'Step {step}: {str({k: round(v, 2) for k, v in results.items()})}')

    return model

def push_to_hub(model):
    hf_repo = f'mrochk/{Pmodel.name}'
    model.push_to_hub(hf_repo)
    get_tokenizer().push_to_hub(hf_repo)

def main():
    torch.manual_seed(Pmain.seed)
    torch.set_float32_matmul_precision('high')

    run = wandb.init(entity='textvsdna', project='discriminators', config=config)

    print('Training discriminator...')
    pp(config)

    dataset = load_generated_dataset()
    print(dataset)

    model = Discriminator()
    print(f'{model}\n#parameters = {count_parameters(model):,}.')
    model = torch.compile(model, mode='max-autotune')

    optim = torch.optim.AdamW(model.parameters())
    model = train_on_split(model, dataset, optim, 0.5, run)

    run.finish()
    push_to_hub(model)

if __name__ == '__main__': main()
