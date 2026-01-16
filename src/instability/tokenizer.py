from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import Whitespace

UNK_TOKEN  = '<unk>'
PAD_TOKEN  = '<pad>'
CLS_TOKEN  = '<cls>'
SEP_TOKEN  = '<sep>'
MASK_TOKEN = '<mask>'
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

def train_bpe_tokenizer(iterator, vocab_size: int) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train_from_iterator(iterator(), trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
    )

    return fast_tokenizer
