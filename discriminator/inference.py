import torch
from tqdm import tqdm
from transformers import set_seed
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count
from datasets import load_from_disk

from discriminator.config import *
from discriminator.model import Discriminator
from discriminator.data import get_real
from discriminator.tokenizer import get_tokenizer, make_preprocess

def compute_weights(logits, T):
    scaled = torch.sigmoid(logits / T)
    return scaled / scaled.mean()

def cut(sample):
    sample['text'] = sample['text'][:Pmain.length]
    return sample

@torch.inference_mode()
def main():
    torch.manual_seed(Pmain.seed)
    set_seed(Pmain.seed)

    hf_repo = f'mrochk/{Pmodel.name}'
    model = Discriminator.from_pretrained(hf_repo).eval().cuda()
    print(model)

    start = 0
    N = 2_000_000 + 200_000

    dataset = get_real().shuffle().select(range(start, start + N)).map(cut)

    local_path = '.cache/inference'

    try: dataset_pp = load_from_disk(local_path)
    except FileNotFoundError:
        pp = make_preprocess(get_tokenizer())
        dataset_pp = dataset.map(pp, num_proc=cpu_count())
        dataset_pp.save_to_disk(local_path)

    print(dataset_pp)
    dataset_pp.set_format('torch')
    dataloader = DataLoader(dataset_pp, batch_size=2048, num_workers=cpu_count(), shuffle=False, pin_memory=True)

    all_logits = []

    for i, batch in enumerate(tqdm(dataloader)):
        inputs = batch['input_ids'].cuda()
        logits = model(inputs)
        all_logits.extend(logits)

    all_logits = torch.tensor(all_logits)
    weights = compute_weights(all_logits, T=1)

    dataset = dataset.add_column('weight', weights.tolist())
    dataset.save_to_disk(f'.cache/weighted_dataset-{Pmodel.name}')
    dataset.push_to_hub(f'mrochk/opengenome-clean-weighted-{Pmodel.name}')

if __name__ == '__main__': main()
