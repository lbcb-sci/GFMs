import torch
import pathlib
import logging

from src.datasets import (
    genomic_benchmarks,
    nt_tasks,
)

def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    return logger

def get_data_path():
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_models_path():
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent / 'models'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_raw_data_path() -> pathlib.Path: 
    data_dir = get_data_path() / 'raw'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_dist_data_path() -> pathlib.Path: 
    data_dir = get_data_path() / 'dist'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_plots_path() -> pathlib.Path: 
    data_dir = get_data_path().parent / 'plots'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_results_path() -> pathlib.Path: 
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent / 'results'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_dl(task: str, batch_size: int, num_workers: int, split: str = 'train'):
    func = genomic_benchmarks.get_dataloader if task in genomic_benchmarks.TASKS else nt_tasks.get_dataloader
    return func(
        task, 
        split=split,
        batch_size=batch_size, 
        num_workers=num_workers,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
