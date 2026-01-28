import re
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, AutoTokenizer

_UNK = '<UNK>'; _PAD = '<PAD>'; _CLS = '<CLS>'; _SEP = '<SEP>'; _MASK = '<MASK>'
_SPECIAL_TOKENS = [_UNK, _PAD, _CLS, _SEP, _MASK]

def train_bpe_tokenizer(iterator, vocab_size: int) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token=_UNK))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_SPECIAL_TOKENS,
    )

    tokenizer.train_from_iterator(iterator(), trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=_UNK,
        pad_token=_PAD,
        cls_token=_CLS,
        sep_token=_SEP,
        mask_token=_MASK,
    )

    return fast_tokenizer

def load_6mer_tokenizer():
    '''Load 6-mer tokenizer from Nucleotide Transformer.'''
    return AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-2.5b-multi-species')

_ALLOWED = r"[^a-zA-Z0-9\s.,;:!?\"'()\-–—/\\&%$€@#\[\]{}<>]+"

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(_ALLOWED, " ", s)
    return s.strip()

def make_iterator(dataset: Dataset):
    def iterator():
        for example in dataset:
            text = example["text"]
            if not isinstance(text, str): continue
            text = clean_text(text)
            if text.strip(): yield text

    return iterator
