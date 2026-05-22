"""
One-time preprocessing script for fasta.gz data.
Walks a root directory, finds one fasta.gz per subdirectory,
extracts all sequences, and splits them into fixed-length chunks.
Chunks shorter than chunk_size are discarded.
"""
import gzip
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from Bio import SeqIO

from src.train.tokenizer import clean_dna

chunk_size = 512 * 8  # characters per chunk

root_path = Path('scratch/ensembl_cdna')  # <-- set this
out_path  = Path('scratch/ensembl_cdna_chunks_4096')  # <-- set this
out_path.mkdir(exist_ok=True)


chunks = []
fasta_files = sorted(root_path.glob('*/*.gz'))

with tqdm(total=len(fasta_files), desc='files') as file_pbar:
    for fasta_path in fasta_files:
        with gzip.open(fasta_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq = clean_dna(str(record.seq)).upper()
                if not seq:
                    continue
                for i in range(0, len(seq), chunk_size):
                    chunk = seq[i : i + chunk_size]
                    if len(chunk) == chunk_size:
                        chunks.append(chunk)
        file_pbar.update(1)

dataset = Dataset.from_dict({'text': chunks})
dataset.save_to_disk(out_path)
print(f'saved {len(chunks)} chunks to {out_path}')
