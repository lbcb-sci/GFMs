from torch.utils.data import DataLoader
from datasets import load_dataset

def get_dataset(split: str = 'train'):
    assert split in ['train', 'test']
    dataset = load_dataset('arcinstitute/opengenome2', split=split, streaming=True)
    return dataset

def get_dataloader(split: str = 'train', batch_size: int = 8):
    dataset = get_dataset(split)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=batch_size,
        shuffle=False,
    )
    return dataloader
