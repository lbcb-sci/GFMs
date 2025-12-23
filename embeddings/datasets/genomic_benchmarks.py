from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.dataset_getters import pytorch_datasets

# datasets that we will use (non-demo and non-dummy)
DATASETS_OF_INTEREST = [
    'human_enhancers_cohn',
    'human_enhancers_ensembl',
    'human_ensembl_regulatory',
    'human_nontata_promoters',
    'human_ocr_ensembl',
]

def get_dataset(dataset: str, split: str = 'train'):
    assert dataset in DATASETS_OF_INTEREST
    assert split in ['train', 'test']
    download_dataset(dataset, version=0)
    return pytorch_datasets.get_dataset(dataset, split=split)