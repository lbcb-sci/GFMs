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
    path = PATHS['savedir']
    path.mkdir(exist_ok=True)
    return path


def get_runs_path() -> Path:
    path = get_savedir_path() / 'runs'
    path.mkdir(exist_ok=True)
    return path


def get_run_path(run_key: str, description: str = None) -> Path:
    time_start = datetime.now().strftime('%y-%m-%d_%H%M%S')
    description = f'_{description}' if description else ''
    run_path = get_runs_path() / f'{time_start}_{run_key}{description}'
    run_path.mkdir(exist_ok=False)
    return run_path


def get_plots_path() -> Path:
    data_dir = get_savedir_path() / 'plots'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_plot_stem(description: str = None) -> str:
    timestamp = datetime.now().strftime('%y-%m-%d_%H%M%S')
    return f'{timestamp}_{description}' if description else timestamp


def get_cache_path() -> Path:
    data_dir = get_savedir_path() / 'cache'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

DATA_TOKENIZER_PAIRS = [
    ('text', 'bpe', 'wiki'),
    ('dna', 'bpe',  'OG2'),
    ('dna', 'kmer', 'OG2'),
    ('dna', 'bpe',  'ncRNA'),
    ('dna', 'kmer', 'ncRNA'),
    ('dna', 'bpe',  'cDNA'),
    ('dna', 'kmer', 'cDNA')
]

def run_key(data: str, tok: str, type: str) -> str:
    return f'{data}_{tok}_{type}' if type else f'{data}_{tok}'

def create_results_dict() -> dict:
    return {run_key(data, tok, type): {} for data, tok, type in DATA_TOKENIZER_PAIRS}