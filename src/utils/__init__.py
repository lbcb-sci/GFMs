import torch
import pathlib
import logging

def get_logger(name: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    return logger

def get_raw_data_folder() -> pathlib.Path: 
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent / 'data' / 'raw'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_dist_data_folder() -> pathlib.Path: 
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent / 'data' / 'dist'
    data_dir.mkdir(exist_ok=True)
    return data_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
