import re
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

_UNK_TOKEN = '<UNK>'; PAD_TOKEN = '<PAD>'; CLS_TOKEN = '<CLS>'; SEP_TOKEN = '<SEP>'; MASK_TOKEN = '<MASK>'
_SPECIAL_TOKENS = [_UNK_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

def train_bpe_tokenizer(iterator, vocab_size: int) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token=_UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_SPECIAL_TOKENS,
    )

    tokenizer.train_from_iterator(iterator(), trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=_UNK_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
    )

    return fast_tokenizer

_ALLOWED = r"[^a-zA-Z0-9\s.,;:!?\"'()\-–—/\\&%$€@#\[\]{}<>]+"

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(_ALLOWED, " ", s)
    return s.strip()

def make_iterator(dataset):
    def iterator():
        for example in dataset:
            text = example["text"]
            if not isinstance(text, str): continue
            text = clean_text(text)
            if text.strip(): yield text
    return iterator
