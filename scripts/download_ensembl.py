"""
Download Ensembl FASTA files for all species, deduplicate species variants,
chop sequences into chunks, and save as a HuggingFace dataset.

Usage:
    python scripts/download_ensembl.py --type cdna --output-dir data/ensembl/cdna --dataset-output data/datasets/cdna
    python scripts/download_ensembl.py --type cds --output-dir data/ensembl/cds --dataset-output data/datasets/cds --chunk-size 512 --seed 42
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import gzip
import json
import random
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL     = 'https://ftp.ensembl.org/pub/release-{release}/fasta/'
SEQUENCE_ALPHABET = re.compile(r'[^ACGTN]')

DATA_TYPE_SUFFIX = {
    'ncrna': '.ncrna.fa.gz',
    'cdna':  '.cdna.all.fa.gz',
    'cds':   '.cds.all.fa.gz',
    'dna':   '.dna.toplevel.fa.gz',
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != 'a':
            return
        for key, value in attrs:
            if key == 'href' and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class DownloadTask:
    species: str
    url: str
    output_path: Path


@dataclass(frozen=True)
class DownloadResult:
    species: str
    url: str
    output_path: str
    status: str
    bytes_written: int
    error: str | None = None


def fetch_text(url: str, timeout: int) -> str:
    request = Request(url, headers={'User-Agent': 'GFMs Ensembl downloader'})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode('utf-8')


def extract_links(html: str) -> list[str]:
    parser = LinkParser()
    parser.feed(html)
    return parser.hrefs


def list_species(base_url: str, timeout: int) -> list[str]:
    html = fetch_text(base_url, timeout)
    species = []
    for href in extract_links(html):
        if href in {'../', '/'} or not href.endswith('/'):
            continue
        name = href[:-1]
        if name:
            species.append(name)
    return sorted(set(species))


def filter_main_species(species_list: list[str]) -> tuple[list[str], list[str]]:
    """Keep only the base species name when versioned duplicates exist.

    e.g. given [ovis_aries, ovis_aries_gca011170295v1, ovis_aries_gca018804185v1]
    keeps ovis_aries and discards the other two.
    """
    sorted_species = sorted(species_list, key=len)  # shortest first
    kept: list[str] = []
    discarded: list[str] = []
    for sp in sorted_species:
        if any(sp.startswith(k) and sp != k for k in kept):
            discarded.append(sp)
        else:
            kept.append(sp)
    return kept, discarded


def list_data_urls(base_url: str, species: str, data_type: str, suffix: str, timeout: int) -> list[str]:
    type_url = urljoin(base_url, f'{species}/{data_type}/')
    try:
        html = fetch_text(type_url, timeout)
    except HTTPError as error:
        if error.code == 404:
            return []
        raise

    urls = []
    for href in extract_links(html):
        if href.endswith(suffix):
            urls.append(urljoin(type_url, href))
    return sorted(set(urls))


def discover_downloads(
    base_url: str,
    output_dir: Path,
    data_type: str,
    suffix: str,
    species_filter: list[str] | None,
    timeout: int,
    workers: int,
    dedup: bool = True,
) -> tuple[list[DownloadTask], list[str], list[str]]:
    all_species = species_filter or list_species(base_url, timeout)

    if dedup:
        kept, discarded = filter_main_species(all_species)
        for sp in discarded:
            print(f'  DISCARD (duplicate) {sp}')
    else:
        kept, discarded = all_species, []

    for sp in kept:
        print(f'  KEEP               {sp}')

    tasks: list[DownloadTask] = []
    missing: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(list_data_urls, base_url, sp, data_type, suffix, timeout): sp
            for sp in kept
        }
        for future in concurrent.futures.as_completed(futures):
            species_name = futures[future]
            urls = future.result()
            if not urls:
                missing.append(species_name)
                continue
            for url in urls:
                filename = Path(url).name
                tasks.append(DownloadTask(
                    species=species_name,
                    url=url,
                    output_path=output_dir / species_name / filename,
                ))

    return tasks, missing, discarded


def fmt_size(n_bytes: int) -> str:
    for unit in ('B', 'K', 'M', 'G'):
        if n_bytes < 1024:
            return f'{n_bytes:.0f}{unit}'
        n_bytes /= 1024
    return f'{n_bytes:.0f}T'


def download_file(task: DownloadTask, overwrite: bool, timeout: int) -> DownloadResult:
    task.output_path.parent.mkdir(parents=True, exist_ok=True)

    if task.output_path.exists() and not overwrite:
        return DownloadResult(
            species=task.species,
            url=task.url,
            output_path=str(task.output_path),
            status='skipped',
            bytes_written=task.output_path.stat().st_size,
        )

    tmp_path = task.output_path.with_suffix(task.output_path.suffix + '.part')
    request = Request(task.url, headers={'User-Agent': 'GFMs Ensembl downloader'})

    try:
        with urlopen(request, timeout=timeout) as response, tmp_path.open('wb') as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(task.output_path)
        return DownloadResult(
            species=task.species,
            url=task.url,
            output_path=str(task.output_path),
            status='downloaded',
            bytes_written=task.output_path.stat().st_size,
        )
    except Exception as error:
        if tmp_path.exists():
            tmp_path.unlink()
        return DownloadResult(
            species=task.species,
            url=task.url,
            output_path=str(task.output_path),
            status='failed',
            bytes_written=0,
            error=str(error),
        )


def normalize_sequence(sequence: str) -> str:
    return SEQUENCE_ALPHABET.sub('N', sequence.upper())


def iter_fasta_sequences(path: Path):
    open_fn = gzip.open if str(path).endswith('.gz') else open
    parts: list[str] = []
    with open_fn(path, 'rt') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if parts:
                    yield ''.join(parts)
                    parts.clear()
            else:
                parts.append(line)
    if parts:
        yield ''.join(parts)


def iter_chunks(sequence: str, chunk_size: int, keep_partial: bool):
    for start in range(0, len(sequence), chunk_size):
        chunk = sequence[start:start + chunk_size]
        if len(chunk) == chunk_size or (keep_partial and chunk):
            yield chunk


def build_dataset(
    file_paths: list[Path],
    dataset_output: Path,
    chunk_size: int,
    min_length: int,
    keep_partial: bool,
    seed: int,
    overwrite: bool,
) -> int:
    from datasets import Dataset, Features, Value
    from datasets.arrow_writer import ArrowWriter

    if not file_paths:
        raise ValueError('No FASTA files found.')

    if dataset_output.exists():
        if not overwrite:
            raise FileExistsError(f'{dataset_output} already exists. Use --overwrite to rebuild.')
        shutil.rmtree(dataset_output)

    dataset_output.parent.mkdir(parents=True, exist_ok=True)

    features  = Features({'text': Value('string')})
    batch_size = 10_000
    rows_written = 0

    with tempfile.TemporaryDirectory(prefix=f'.{dataset_output.name}_', dir=dataset_output.parent) as tmpdir:
        arrow_path = Path(tmpdir) / 'data.arrow'
        writer = ArrowWriter(path=str(arrow_path), features=features)

        try:
            batch: list[str] = []
            for path in file_paths:
                for seq in iter_fasta_sequences(path):
                    seq = normalize_sequence(seq)
                    for chunk in iter_chunks(seq, chunk_size, keep_partial):
                        if len(chunk) < min_length:
                            continue
                        batch.append(chunk)
                        if len(batch) >= batch_size:
                            writer.write_batch({'text': batch})
                            rows_written += len(batch)
                            batch = []
            if batch:
                writer.write_batch({'text': batch})
                rows_written += len(batch)
            writer.finalize()

            dataset = Dataset.from_file(str(arrow_path))
            if seed is not None:
                print(f'  Written {rows_written:,} chunks, shuffling with seed={seed} ...')
                dataset = dataset.shuffle(seed=seed)
            else:
                print(f'  Written {rows_written:,} chunks, no shuffling.')
            dataset.save_to_disk(dataset_output)
            del dataset
            gc.collect()
        finally:
            writer.close()

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download Ensembl FASTA files and build a HuggingFace dataset.')
    parser.add_argument('--type',           type=str, required=True, choices=list(DATA_TYPE_SUFFIX), help='Data type to download.')
    parser.add_argument('--release',        type=int, default=115)
    parser.add_argument('--output-dir',     type=Path, required=True, help='Directory to store downloaded FASTA files.')
    parser.add_argument('--dataset-output', type=Path, default=None, help='Path to save the HuggingFace dataset.')
    parser.add_argument('--chunk-size',     type=int, default=4096)
    parser.add_argument('--min-length',     type=int, default=None, help='Discard chunks shorter than this. Defaults to chunk-size unless --keep-partial is set.')
    parser.add_argument('--keep-partial',   action='store_true', help='Keep trailing chunks shorter than chunk-size.')
    parser.add_argument('--seed',           type=int, default=None, help='Random seed for shuffling. If not set, dataset is not shuffled.')
    parser.add_argument('--workers',        type=int, default=8)
    parser.add_argument('--timeout',        type=int, default=120)
    parser.add_argument('--species',        nargs='*', default=None, help='Explicit species to download (skips deduplication).')
    parser.add_argument('--no-dedup',       action='store_true', help='Disable species deduplication; download all variants.')
    parser.add_argument('--local-only',     action='store_true', help='Skip download; build dataset from already-downloaded files.')
    parser.add_argument('--overwrite',      action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    suffix     = DATA_TYPE_SUFFIX[args.type]
    base_url   = BASE_URL.format(release=args.release)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    min_length = args.min_length if args.min_length is not None else (1 if args.keep_partial else args.chunk_size)

    results: list[DownloadResult] = []

    if args.local_only:
        all_file_paths = sorted(output_dir.glob(f'**/*{suffix}'))
        if not all_file_paths:
            raise RuntimeError(f'No {args.type} FASTA files found under {output_dir}.')
        print(f'Found {len(all_file_paths)} local files under {output_dir}.')
    else:
        print(f'Listing species from {base_url} ...')
        tasks, missing, discarded = discover_downloads(base_url, output_dir, args.type, suffix, args.species, args.timeout, args.workers, dedup=not args.no_dedup)

        if not tasks:
            raise RuntimeError(f'No {args.type} files discovered for release {args.release}.')

        n_species = len({t.species for t in tasks})
        print(f'\nDiscovered {len(tasks)} files across {n_species} species ({len(discarded)} duplicates discarded).')
        if missing:
            print(f'{len(missing)} species had no {args.type} directory: {missing}')

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_file, task, args.overwrite, args.timeout) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                filename = Path(result.output_path).name
                size     = fmt_size(result.bytes_written)
                if result.status == 'downloaded':
                    print(f'  [OK]   {result.species}: {filename} ({size})')
                elif result.status == 'skipped':
                    print(f'  [SKIP] {result.species}: {filename} ({size})')
                elif result.status == 'failed':
                    print(f'  [FAIL] {result.species}: {result.error}')

        all_file_paths = [Path(r.output_path) for r in results if r.status in {'downloaded', 'skipped'}]

    summary = {
        'release': args.release,
        'type': args.type,
        'chunk_size': args.chunk_size,
        'min_length': min_length,
        'keep_partial': args.keep_partial,
        'seed': args.seed,
        'downloaded': sum(r.status == 'downloaded' for r in results),
        'skipped': sum(r.status == 'skipped' for r in results),
        'failed': [asdict(r) for r in results if r.status == 'failed'],
    }

    if args.dataset_output is not None:
        print(f'\nBuilding dataset from {len(all_file_paths)} files ...')
        n_rows = build_dataset(
            all_file_paths,
            args.dataset_output,
            args.chunk_size,
            min_length,
            args.keep_partial,
            args.seed,
            args.overwrite,
        )
        summary['dataset_rows'] = n_rows
        print(f'Saved dataset with {n_rows:,} rows to {args.dataset_output}.')

    summary_path = output_dir / f'release_{args.release}_{args.type}_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'Summary written to {summary_path}.')

    if summary['failed']:
        raise RuntimeError(f'{len(summary["failed"])} downloads failed. See {summary_path}.')


if __name__ == '__main__':
    main()
