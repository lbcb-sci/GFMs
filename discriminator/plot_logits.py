import torch
import transformers
from tqdm import tqdm
from math import sqrt
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count

import seaborn as sns
import matplotlib.pyplot as plt

from discriminator.config import *
from discriminator.model import Discriminator
from discriminator.tokenizer import get_tokenizer, make_preprocess
from discriminator.data import get_real

@torch.inference_mode()
def collect(model, dataloader, n):
    all_labels = []; all_logits = []; all_probs = []

    for i, batch in enumerate(tqdm(dataloader, total=n)):
        sequences = batch['input_ids'].cuda()
        #labels = batch['real']
        labels = torch.ones(len(sequences))

        logits = model(sequences)
        probs = torch.sigmoid(logits)

        all_labels.extend(labels.cpu().tolist())
        all_logits.extend(logits.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        if n is not None and (i+1) == n: break

    labels = torch.tensor(all_labels).int()
    logits = torch.tensor(all_logits)
    probs  = torch.tensor(all_probs)

    return labels, logits, probs

def compute_weights(logits, T):
    scaled = torch.sigmoid(logits / T)
    return scaled / scaled.mean()

def effective_sample_size(weights):
    N = len(weights)
    neff = N**2 / (weights**2).sum()
    return neff / N

def expected_coverage(weights):
    N = len(weights)
    num_samples = N
    probs = weights / weights.sum()
    p_seen = 1 - (1 - probs) ** num_samples
    return p_seen.mean().item()

def cut(sample):
    sample['text'] = sample['text'][:Pmain.length]
    return sample

def main():
    sns.set_theme('paper', style='whitegrid', palette='pastel')

    torch.manual_seed(Pmain.seed)
    transformers.set_seed(Pmain.seed)

    model_path = f'mrochk/{Pmodel.name}'
    model = Discriminator.from_pretrained(model_path).eval().cuda(); print(model)
    model = torch.compile(model, mode='max-autotune')

    tokenizer = get_tokenizer()

    dataset = get_real(1_000_000).select(range(900_000, 1_000_000)).map(cut); print(dataset)
    dataset = dataset.map(make_preprocess(tokenizer), num_proc=cpu_count()//3, remove_columns=['text'])
    dataset.set_format('torch')
    dataloader = DataLoader(dataset, Ptrain.batch_size, shuffle=True)

    n = 10 #None
    labels, logits, probs = collect(model, dataloader, n)

    for T in [0.5, 1, 2, 5]:

        weights = compute_weights(logits, T)

        print('eff:', torch.sum(weights) / len(weights))

        nbins = int(sqrt(len(weights)))
        plt.hist(weights, bins=nbins, color='blue')
        plt.savefig(f'plots/dist_weights_{T}.png')
        plt.close()

        print(weights)
        print(effective_sample_size(weights))
        print(expected_coverage(weights))

    # plot distributions
    nbins = int(sqrt(len(probs)))

    plt.hist(probs, bins=nbins, color='orange')
    plt.savefig('plots/dist_probs.png')
    plt.close()

    plt.hist(logits, bins=nbins, color='black')
    plt.savefig('plots/dist_logits.png')
    plt.close()

    #for beta in [1, 1.5, 2]:
        #weights = F.softplus(real_logits, beta=beta)
        #plt.hist(weights, bins=nbins, color='royalblue')
        #plt.savefig(f'plots/dist_weights_beta{beta}.png')
        #plt.close()

if __name__ == '__main__': main()
