import time
import pathlib
import logging
from torch import Tensor
from transformers import BertModel, BertForMaskedLM

def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    return logger

def get_runs_path():
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent / 'runs'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def make_run_path(type: str, tokenizer: str) -> pathlib.Path:
    runs_path = get_runs_path()
    run_path = runs_path / f'{int(time.time())}_{type}_{tokenizer}'
    run_path.mkdir(exist_ok=False)
    return run_path

def get_plots_path() -> pathlib.Path: 
    data_dir = get_runs_path().parent / 'plots'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(path) -> BertModel:
    return BertForMaskedLM.from_pretrained(str(path.resolve()), local_files_only=True).eval().bert

def load_embeddings(path) -> Tensor:
    model = load_model(path)
    return model.embeddings.word_embeddings.weight.detach()

def load_many_embeddings(paths: list) -> list[Tensor]:
    return [load_embeddings(path) for path in paths]
