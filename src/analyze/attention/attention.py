from logging import Logger
from transformers import PreTrainedTokenizer

def analyze_attention(
    models_dict: dict, 
    tokenizer: PreTrainedTokenizer, 
    logger: Logger,
    n_samples: int,
    batch_size: int,
    p_mask: float = 0.15,
) -> dict:
    pass