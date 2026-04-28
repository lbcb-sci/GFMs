from __future__ import annotations

import argparse
import concurrent.futures
import gc
import gzip
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = 'https://ftp.ensembl.org/pub/release-{release}/fasta/'
NCRNA_SUFFIX = '.ncrna.fa.gz'
SEQUENCE_ALPHABET = re.compile(r'[^ACGTN]')


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
    request = Request(url, headers={'User-Agent': 'GFMs ncRNA downloader'})
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
        if not name:
            continue
        species.append(name)

    return sorted(set(species))


def list_ncrna_urls(base_url: str, species: str, timeout: int) -> list[str]:
    ncrna_url = urljoin(base_url, f'{species}/ncrna/')

    try:
        html = fetch_text(ncrna_url, timeout)
    except HTTPError as error:
        if error.code == 404:
            return []
        raise
    except URLError:
        raise

    urls = []
    for href in extract_links(html):
        if href.endswith(NCRNA_SUFFIX):
            urls.append(urljoin(ncrna_url, href))

    return sorted(set(urls))


def discover_downloads(base_url: str, output_dir: Path, species_filter: list[str] | None, timeout: int) -> tuple[list[DownloadTask], list[str]]:
    species = species_filter or list_species(base_url, timeout)

    tasks: list[DownloadTask] = []
    missing_species: list[str] = []

    for species_name in species:
        urls = list_ncrna_urls(base_url, species_name, timeout)
        if not urls:
            missing_species.append(species_name)
            continue

        for url in urls:
            filename = Path(url).name
            tasks.append(
                DownloadTask(
                    species=species_name,
                    url=url,
                    output_path=output_dir / species_name / filename,
                )
            )

    return tasks, missing_species


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
    request = Request(task.url, headers={'User-Agent': 'GFMs ncRNA downloader'})

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


def clean_sequence(sequence: str) -> str:
    try:
        from src.train.tokenizer import clean_dna
    except ModuleNotFoundError:
        return normalize_sequence(sequence)
    return clean_dna(sequence).upper()


def discover_local_files(output_dir: Path, species_filter: list[str] | None = None) -> list[Path]:
    if species_filter:
        result = []
        for species in species_filter:
            species_dir = output_dir / species
            result.extend(sorted(species_dir.glob(f'*{NCRNA_SUFFIX}')))
        return [path for path in result if path.is_file()]

    return sorted(path for path in output_dir.glob(f'**/*{NCRNA_SUFFIX}') if path.is_file())


def iter_fasta_sequences(path: Path):
    sequence_parts: list[str] = []

    with gzip.open(path, 'rt') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence_parts:
                    yield ''.join(sequence_parts)
                    sequence_parts.clear()
                continue
            sequence_parts.append(line)

    if sequence_parts:
        yield ''.join(sequence_parts)


def iter_chunks(sequence: str, chunk_size: int, keep_partial_chunks: bool):
    for start in range(0, len(sequence), chunk_size):
        chunk = sequence[start:start + chunk_size]
        if len(chunk) == chunk_size or (keep_partial_chunks and chunk):
            yield chunk


def iter_chunk_texts(file_paths: list[str], chunk_size: int, min_length: int, keep_partial_chunks: bool):
    for file_path in file_paths:
        for sequence in iter_fasta_sequences(Path(file_path)):
            cleaned = clean_sequence(sequence)
            for chunk in iter_chunks(cleaned, chunk_size, keep_partial_chunks):
                if len(chunk) >= min_length:
                    yield chunk


def count_chunk_lengths(file_paths: list[str], chunk_size: int, min_length: int, keep_partial_chunks: bool) -> tuple[list[int], int]:
    counts = [0] * (chunk_size + 1)
    total = 0

    for chunk in iter_chunk_texts(file_paths, chunk_size, min_length, keep_partial_chunks):
        counts[len(chunk)] += 1
        total += 1

    return counts, total


def plan_chunk_selection(counts: list[int], chunk_size: int, max_sequences: int | None) -> dict:
    total_available = sum(counts)
    full_length_count = counts[chunk_size]

    if max_sequences is None or max_sequences >= total_available:
        return {
            'max_sequences': max_sequences,
            'total_available': total_available,
            'selected_rows': total_available,
            'full_length_count': full_length_count,
            'use_all': True,
            'cutoff_length': None,
            'take_at_cutoff': None,
        }

    if max_sequences <= 0:
        raise ValueError('--max-sequences must be a positive integer.')

    if max_sequences < full_length_count:
        raise ValueError(
            f'--max-sequences={max_sequences:,} is smaller than the {full_length_count:,} full-length chunks, '
            'so the requested selection rule cannot be satisfied.'
        )

    kept_so_far = 0
    for length in range(chunk_size, 0, -1):
        count = counts[length]
        if count == 0:
            continue

        if kept_so_far + count >= max_sequences:
            return {
                'max_sequences': max_sequences,
                'total_available': total_available,
                'selected_rows': max_sequences,
                'full_length_count': full_length_count,
                'use_all': False,
                'cutoff_length': length,
                'take_at_cutoff': max_sequences - kept_so_far,
            }

        kept_so_far += count

    raise RuntimeError('Failed to construct a chunk selection plan.')


def generate_dataset_rows(
    file_paths: list[str],
    chunk_size: int,
    min_length: int,
    keep_partial_chunks: bool,
    selection_plan: dict | None = None,
):
    use_all = selection_plan is None or selection_plan['use_all']
    cutoff_length = None if use_all else selection_plan['cutoff_length']
    remaining_at_cutoff = None if use_all else selection_plan['take_at_cutoff']

    for chunk in iter_chunk_texts(file_paths, chunk_size, min_length, keep_partial_chunks):
        if use_all or len(chunk) > cutoff_length:
            yield {'text': chunk}
            continue

        if len(chunk) == cutoff_length and remaining_at_cutoff and remaining_at_cutoff > 0:
            yield {'text': chunk}
            remaining_at_cutoff -= 1


def build_dataset(
    file_paths: list[Path],
    dataset_output: Path,
    chunk_size: int,
    min_length: int,
    keep_partial_chunks: bool,
    max_sequences: int | None,
    overwrite: bool,
) -> tuple[int, list[str], dict]:
    from datasets import Dataset, Features, Value
    from datasets.arrow_writer import ArrowWriter

    if not file_paths:
        raise ValueError('No ncRNA FASTA files were selected for dataset creation.')

    if dataset_output.exists():
        if not overwrite:
            raise FileExistsError(f'{dataset_output} already exists. Use --overwrite to rebuild it.')
        shutil.rmtree(dataset_output)

    dataset_output.parent.mkdir(parents=True, exist_ok=True)

    features = Features({'text': Value('string')})
    batch_size = 10_000
    rows_written = 0
    file_path_strings = [str(path) for path in file_paths]
    counts, _ = count_chunk_lengths(file_path_strings, chunk_size, min_length, keep_partial_chunks)
    selection_plan = plan_chunk_selection(counts, chunk_size, max_sequences)

    with tempfile.TemporaryDirectory(prefix=f'.{dataset_output.name}_', dir=dataset_output.parent) as tmpdir:
        arrow_path = Path(tmpdir) / 'data.arrow'
        writer = ArrowWriter(path=str(arrow_path), features=features)

        try:
            batch: list[str] = []
            for row in generate_dataset_rows(
                file_path_strings,
                chunk_size,
                min_length,
                keep_partial_chunks,
                selection_plan,
            ):
                batch.append(row['text'])
                if len(batch) >= batch_size:
                    writer.write_batch({'text': batch})
                    rows_written += len(batch)
                    batch = []

            if batch:
                writer.write_batch({'text': batch})
                rows_written += len(batch)

            writer.finalize()
            dataset = Dataset.from_file(str(arrow_path))
            dataset.save_to_disk(dataset_output)
            del dataset
            gc.collect()
        finally:
            writer.close()

    return rows_written, file_path_strings, selection_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download Ensembl ncRNA FASTA files for all species in one release.')
    parser.add_argument('--release', type=int, default=115, help='Ensembl release number to download from.')
    parser.add_argument('--output-dir', type=Path, default=Path('data/ensembl/release-115/ncrna/raw'), help='Directory where compressed FASTA files will be stored.')
    parser.add_argument('--dataset-output', type=Path, default=None, help='Optional output path for a HuggingFace dataset saved to disk with a single text column.')
    parser.add_argument('--chunk-size', type=int, default=512 * 8, help='Maximum number of nucleotides per saved text example. Defaults to 4096 to match the cDNA preprocessing used in the repo.')
    parser.add_argument('--min-length', type=int, default=None, help='Discard chunks shorter than this many nucleotides when building the dataset. Defaults to chunk-size unless --keep-partial-chunks is set.')
    parser.add_argument('--workers', type=int, default=8, help='Number of concurrent downloads.')
    parser.add_argument('--timeout', type=int, default=120, help='Per-request timeout in seconds.')
    parser.add_argument('--species', nargs='*', default=None, help='Optional explicit species directory names to download.')
    parser.add_argument('--local-only', action='store_true', help='Skip remote discovery/download and operate on the files already present in output-dir.')
    parser.add_argument('--keep-partial-chunks', action='store_true', help='Keep trailing chunks shorter than chunk-size. By default they are dropped to match the existing cDNA preprocessing.')
    parser.add_argument('--max-sequences', type=int, default=None, help='Optional cap on how many chunks to save. Full-length chunks are always kept first, then shorter chunks with the least padding are included until the cap is reached.')
    parser.add_argument('--overwrite', action='store_true', help='Redownload existing files and rebuild an existing dataset output path.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = BASE_URL.format(release=args.release)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[DownloadResult] = []
    missing_species: list[str] = []

    if args.local_only:
        all_file_paths = discover_local_files(output_dir, args.species)
        if not all_file_paths:
            raise RuntimeError(f'No ncRNA FASTA files found under {output_dir}.')
        print(f'Found {len(all_file_paths)} local ncRNA FASTA files under {output_dir}.')
    else:
        tasks, missing_species = discover_downloads(base_url, output_dir, args.species, args.timeout)
        if not tasks:
            raise RuntimeError(f'No ncRNA FASTA files were discovered for release {args.release}.')

        print(f'Discovered {len(tasks)} ncRNA FASTA files across {len({task.species for task in tasks})} species.')
        if missing_species:
            print(f'{len(missing_species)} species had no ncRNA directory or file listing.')

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_file, task, args.overwrite, args.timeout) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                if result.status == 'failed':
                    print(f'FAILED {result.species}: {result.error}')

        results.sort(key=lambda item: (item.species, item.output_path))
        all_file_paths = [Path(result.output_path) for result in results if result.status in {'downloaded', 'skipped'}]

    downloaded = sum(result.status == 'downloaded' for result in results)
    skipped = sum(result.status == 'skipped' for result in results)
    failed = [result for result in results if result.status == 'failed']
    min_length = args.min_length if args.min_length is not None else (1 if args.keep_partial_chunks else args.chunk_size)

    summary = {
        'release': args.release,
        'base_url': base_url,
        'download_root': str(output_dir),
        'local_only': args.local_only,
        'dataset_output': str(args.dataset_output) if args.dataset_output else None,
        'chunk_size': args.chunk_size,
        'min_length': min_length,
        'keep_partial_chunks': args.keep_partial_chunks,
        'max_sequences': args.max_sequences,
        'discovered_files': len(all_file_paths),
        'downloaded_files': downloaded,
        'skipped_files': skipped,
        'failed_files': len(failed),
        'species_without_ncrna': missing_species,
        'results': [asdict(result) for result in results],
    }

    if args.dataset_output is not None and not failed:
        n_rows, file_paths, selection_plan = build_dataset(
            all_file_paths,
            args.dataset_output,
            args.chunk_size,
            min_length,
            args.keep_partial_chunks,
            args.max_sequences,
            args.overwrite,
        )
        summary['dataset_rows'] = n_rows
        summary['dataset_source_files'] = len(file_paths)
        summary['selection_plan'] = selection_plan
        if not selection_plan['use_all']:
            print(
                f"Selected the top {selection_plan['selected_rows']:,} chunks out of {selection_plan['total_available']:,} "
                f"by chunk length. Cutoff length: {selection_plan['cutoff_length']}, "
                f"taking {selection_plan['take_at_cutoff']:,} chunk(s) at that length."
            )
        print(f'Saved dataset with {n_rows:,} rows to {args.dataset_output}.')

    summary_path = output_dir / f'release_{args.release}_download_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'Wrote summary to {summary_path}.')

    if failed:
        raise RuntimeError(f'{len(failed)} downloads failed. See {summary_path} for details.')


if __name__ == '__main__':
    main()