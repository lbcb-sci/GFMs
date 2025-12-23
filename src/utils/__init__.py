import torch
import pathlib

def get_data_folder() -> pathlib.Path: 
    data_dir = pathlib.Path(__file__).resolve().parent.parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    return data_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
