from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from discriminator.config import Pmain 

vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '[UNK]': 5}

def get_tokenizer() -> PreTrainedTokenizerFast:
    '''Creates the simplest possible DNA tokenizer, with mapping [A->0, C->1, G->2, T->3].'''
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Split(pattern='', behavior='isolated')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.model_max_length = Pmain.length
    return tokenizer

def make_preprocess(tokenizer: PreTrainedTokenizerFast):
    def preprocess(batch): return tokenizer(
        batch['text'], 
        truncation=True, 
        return_attention_mask=False,
    )
    return preprocess 
