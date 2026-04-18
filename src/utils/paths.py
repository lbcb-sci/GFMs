from pathlib import Path

_scratch = Path('scratch')
_data    = Path('data')

PATHS = {
    'text_bpe_tokenizer':  _scratch / 'bpe_text_tokenizer',
    'dna_bpe_tokenizer':   _scratch / 'bpe_dna_tokenizer',
    'dna_lcp_tokenizer':   _scratch / 'lcp_dna_tokenizer',
    'checkpoints':         _scratch / 'checkpoints',
    'savedir':             _scratch,
    'og2_dataset':         _data / 'opengenome2_subset/preprocessed_12M_uppercase',
    'wiki_dataset':        _data / 'wikipedia_packed',
    'ensembl_cdna_chunks': _scratch / 'ensembl_cdna_chunks_4096',
    'cache_dir':           _data,
}

try:
    from src.utils.paths_local import PATHS as _local
    PATHS = {**PATHS, **_local}
except ImportError:
    pass
