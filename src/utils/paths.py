from pathlib import Path

_scratch = Path('/home/vrcekl/scratch/GFMs')
_data    = Path('/mnt/sod2-project/csb4/wgs/lovro/huggingface')

PATHS = {
    'text_bpe_tokenizer': _scratch / 'bpe_text_tokenizer',
    'dna_bpe_tokenizer':  _scratch / 'bpe_dna_tokenizer',
    'checkpoints':        _scratch / 'checkpoints',
    'savedir':            _scratch,
    'og2_dataset':        _data / 'opengenome2_subset/preprocessed_12M_uppercase',
    'ensembl_cdna_chunks': _scratch / 'ensembl_cdna_chunks_4096',
    'cache_dir':          _data,
}
