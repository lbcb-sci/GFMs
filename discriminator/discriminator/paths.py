from pathlib import Path

_scratch = Path('scratch')
_data    = Path('data')

PATHS = {
    'ensembl_dataset':     _scratch / 'ensembl_cdna_chunks_4096',
    'username':            'user',
    'save_path':           _data / 'dataset',
}

try:
    from src.utils.paths_local import PATHS as _local
    PATHS = {**PATHS, **_local}
except ImportError:
    pass
