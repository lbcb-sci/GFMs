import torch
from tqdm import tqdm
from pprint import pp
from torch.multiprocessing import cpu_count
from datasets import concatenate_datasets, Dataset, DatasetDict

from discriminator.config import Pmain, Pdata
from discriminator.tokenizer import get_tokenizer
from discriminator.data import get_real, preprocess_dataset, get_cdna

def sample_is_clean(sample):
    return set(sample['text']) == set(['A', 'T', 'C', 'G'])

def cut(sample):
    sample['text'] = sample['text'][:Pmain.length]
    return sample

def main():
    torch.manual_seed(Pmain.seed)

    coding = get_cdna()
    coding = coding.filter(sample_is_clean, num_proc=cpu_count()).map(cut)

    real = get_real(len(coding)).map(cut)

    print(len(coding[0]['text']))
    print(len(real[0]['text']))

    tokenizer = get_tokenizer()
    real = preprocess_dataset(real, tokenizer, False)
    coding = preprocess_dataset(coding, tokenizer, True)

    final = concatenate_datasets([real, coding]).shuffle(Pmain.seed)
    final.set_format('torch')
    final.save_to_disk(Pdata.save_path)

if __name__ == '__main__': main()
