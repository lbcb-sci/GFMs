from time import time
from pathlib import Path
import logging

def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

def get_root_path():
    return Path(__file__).resolve().parent.parent.parent

def get_runs_path():
    path = get_root_path() / 'runs'
    path.mkdir(exist_ok=True)
    return path

def get_run_path(type: str, tokenizer: str) -> Path:
    run_path = get_runs_path() / f'{int(time())}_{type}_{tokenizer}'
    run_path.mkdir(exist_ok=False)
    return run_path

def get_plots_path() -> Path: 
    data_dir = get_root_path() / 'plots'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_cache_path() -> Path: 
    data_dir = get_root_path() / 'cache'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
