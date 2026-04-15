import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

from src.analyze.data import mlm_preprocess, get_opengenome, get_wikipedia, DeviceWrapper
from src.utils import N, DATA_TOKENIZER_PAIRS, create_results_dict

def activations(all_models: dict, tokenizers: dict, args) -> dict: pass