from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

DATASET_NAME = 'InstaDeepAI/nucleotide_transformer_downstream_tasks_revised'

TASKS = [
    'promoter_all', 
    'promoter_tata', 
    'promoter_no_tata', 
    'enhancers', 
    'enhancers_types', 
    'splice_sites_all', 
    'splice_sites_acceptor', 
    'splice_sites_donor', 
    'H2AFZ', 'H3K27ac', 
    'H3K27me3', 'H3K36me3', 
    'H3K4me1', 'H3K4me2', 
    'H3K4me3', 'H3K9ac', 
    'H3K9me3',
]

BINARY_TASKS = TASKS[:4] + TASKS[8:]

class NTDataset(Dataset):
    def __init__(self, ds): self.ds = ds

    def __len__(self): return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        return tuple([sample['sequence'], sample['label']])

def get_dataset(task: str, split: str = 'train'):
    match split:
        case 'train': return NTDataset(get_train_set(task))
        case 'test': return NTDataset(get_test_set(task))
        case _: raise Exception()

def get_dataloader(task: str, split: str, batch_size: int, num_workers: int):
    dataset = get_dataset(task, split)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=True,
    )
    return dataloader

def get_train_set(task: str):
    if task not in TASKS: raise Exception('task not supported')

    dataset_train = load_dataset(
        path=DATASET_NAME,
        split='train',
        streaming=False,
    )

    dataset_train = dataset_train.filter(lambda x: x['task'] == task)
    return dataset_train

def get_test_set(task: str):
    if task not in TASKS: raise Exception('task not supported')
    dataset_test = load_dataset(
        path=DATASET_NAME,
        split='test',
        streaming=False,
    )

    dataset_test = dataset_test.filter(lambda x: x['task'] == task)
    return dataset_test