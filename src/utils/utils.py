import logging
from datetime import datetime
from pathlib import Path

from src.utils.paths import PATHS


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def get_root_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_savedir_path() -> Path:
    path = Path('/root/GFMs')
    #path = PATHS['savedir']
    path.mkdir(exist_ok=True)
    return path


def get_runs_path() -> Path:
    path = get_savedir_path() / 'runs'
    path.mkdir(exist_ok=True)
    return path


def get_run_path(type: str, tokenizer: str, description: str = None) -> Path:
    time_start = datetime.now().strftime('%y-%m-%d_%H%M%S')
    description = f'_{description}' if description else ''
    run_path = get_runs_path() / f'{time_start}_{type}_{tokenizer}{description}'
    run_path.mkdir(exist_ok=False)
    return run_path


def get_plots_path() -> Path:
    data_dir = get_savedir_path() / 'plots'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_cache_path() -> Path:
    data_dir = get_savedir_path() / 'cache'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

DATA_TOKENIZER_PAIRS = [('text', 'bpe'), ('dna', 'bpe'), ('dna', 'kmer')]

def create_results_dict() -> dict: 
    results = {}
    for data, tok in DATA_TOKENIZER_PAIRS:
        if data not in results.keys(): results[data] = {}
        results[data][tok] = {}

    return results