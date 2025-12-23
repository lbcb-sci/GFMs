from torch.utils.data import DataLoader
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.dataset_getters import pytorch_datasets

# datasets that we will use (non-demo and non-dummy)
TASKS = [
    'human_enhancers_cohn',
    'human_enhancers_ensembl',
    'human_ensembl_regulatory',
    'human_nontata_promoters',
    'human_ocr_ensembl',
]

def get_dataset(dataset: str, split: str = 'train'):
    assert dataset in TASKS
    assert split in ['train', 'test']
    download_dataset(dataset, version=0)
    return pytorch_datasets.get_dataset(dataset, split=split)

def get_dataloader(task: str, split: str = 'train', batch_size: int = 8):
    dataset = get_dataset(task, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader