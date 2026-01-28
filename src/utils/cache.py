'''Simplest persistent cache ever.'''

import torch
from torch import Tensor
from os import listdir
from .utils import get_cache_path

def cached(filename: str) -> bool:
    return filename in listdir(get_cache_path())

def store(filename: str, data: dict | Tensor) -> None:
    torch.save(data, get_cache_path() / filename)

def get(filename: str) -> Tensor | dict:
    return torch.load(get_cache_path() / filename)
