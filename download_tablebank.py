"""Parallel download of TableBank from Hugging Face into a Modal volume.

Downloads every file from the dataset repo using chunked byte-range requests
spread across many containers, then reassembles each downloaded file on the
volume. Multipart ZIP sets such as ``TableBank.zip.001`` ... ``.005`` are then
concatenated into a full archive and optionally extracted.

Usage
-----
    modal run download_tablebank.py
    modal run download_tablebank.py --max-workers 50
    modal run download_tablebank.py --output-subdir tablebank_full
    modal run download_tablebank.py --no-include-metadata
    modal run download_tablebank.py --no-extract-archives
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]", "requests")
)

app = modal.App(name="tablebank-download", image=download_image)

dataset_vol = modal.Volume.from_name("tablebank-vol", create_if_missing=True, version=2)
MODAL_DATA_DIR = "/data"

DATASET_ID = "liminghao1630/TableBank"

# Files smaller than this are downloaded in a single request (no chunking).
_CHUNK_THRESHOLD = 500 * 1024 * 1024  # 500 MB

# Modal Volume v2 supports at most 260,000 files per directory.
# TableBank contains very large flat directories (for example Detection/images),
# so extracted files are sharded into 256 hash buckets by default.
_SHARD_HEX_CHARS = 2

_MULTIPART_ZIP_RE = re.compile(r"^(?P<base>.+\.zip)\.(?P<part>\d+)$")


# ---------------------------------------------------------------------------
# Phase 0 - list dataset files
# ---------------------------------------------------------------------------


@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=120,
)
def list_dataset_files(include_metadata: bool = True) -> list[str]:
    """Return all files in the dataset repo."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    filenames = sorted(api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset"))
    if not include_metadata:
        filenames = [
            name for name in filenames
            if name != "README.md" and not Path(name).name.startswith(".")
        ]

    for name in filenames:
        print(name)
    return filenames


# ---------------------------------------------------------------------------
# Phase 1a - probe file sizes via HEAD
# ---------------------------------------------------------------------------


@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=120,
)
def probe_files(filenames: list[str]) -> dict[str, int]:
    """Return ``{filename: size_in_bytes}`` for every file."""
    import requests
    from huggingface_hub import hf_hub_url

    token = os.environ.get("HF_TOKEN", "")
    sizes: dict[str, int] = {}
    for filename in filenames:
        url = hf_hub_url(repo_id=DATASET_ID, filename=filename, repo_type="dataset")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        size = int(resp.headers.get("Content-Length", 0))
        sizes[filename] = size
        print(f"  {filename}: {size / 1e9:.2f} GB")
    return sizes


# ---------------------------------------------------------------------------
# Phase 1b - download one byte-range chunk
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    memory=4096,
    retries=2,
)
def download_chunk(
    filename: str,
    chunk_idx: int,
    start_byte: int,
    end_byte: int,
    chunk_dir: str,
) -> dict[str, Any]:
    """Download bytes ``[start_byte, end_byte]`` to a numbered chunk file."""
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=DATASET_ID, filename=filename, repo_type="dataset")
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    headers["Range"] = f"bytes={start_byte}-{end_byte}"

    Path(chunk_dir).mkdir(parents=True, exist_ok=True)
    chunk_path = Path(chunk_dir) / f"chunk_{chunk_idx:04d}.bin"

    written = 0
    with requests.get(url, headers=headers, stream=True, timeout=(60, 600)) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(
                f"HTTP {resp.status_code} for {filename} "
                f"range {start_byte}-{end_byte}"
            )
        with open(chunk_path, "wb") as fh:
            for data in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(data)
                written += len(data)

    dataset_vol.commit()
    expected = end_byte - start_byte + 1
    print(f"[{filename}] chunk {chunk_idx}: {written:,} / {expected:,} bytes")
    return {"repo_file": filename, "chunk_idx": chunk_idx, "written": written}


# ---------------------------------------------------------------------------
# Phase 1c - reassemble ordered chunks into one file
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=14400,
    memory=4096,
    retries=1,
)
def reassemble_to_file(
    repo_file: str,
    chunk_dir: str,
    num_chunks: int,
    output_path: str,
) -> dict[str, Any]:
    """Reassemble numbered chunks into a single file on the volume."""
    dataset_vol.reload()

    chunk_paths = [Path(chunk_dir) / f"chunk_{i:04d}.bin" for i in range(num_chunks)]
    missing = [path for path in chunk_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing chunks for {repo_file}: {missing}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    with open(destination, "wb") as out_fh:
        for idx, chunk_path in enumerate(chunk_paths, start=1):
            with open(chunk_path, "rb") as in_fh:
                shutil.copyfileobj(in_fh, out_fh, length=8 * 1024 * 1024)
            bytes_written += chunk_path.stat().st_size
            if idx % 4 == 0:
                dataset_vol.commit()

    for path in chunk_paths:
        path.unlink(missing_ok=True)

    dataset_vol.commit()
    print(f"[{repo_file}] wrote {bytes_written / 1e9:.2f} GB to {destination}")
    return {
        "repo_file": repo_file,
        "output_path": str(destination),
        "bytes_written": bytes_written,
    }


# ---------------------------------------------------------------------------
# Phase 2 - concatenate multipart ZIP archives
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=14400,
    memory=4096,
    retries=1,
)
def concatenate_files(
    input_paths: list[str],
    output_path: str,
    delete_inputs: bool = False,
) -> dict[str, Any]:
    """Concatenate multiple files into one archive."""
    dataset_vol.reload()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    with open(destination, "wb") as out_fh:
        for idx, input_path in enumerate(input_paths, start=1):
            source = Path(input_path)
            if not source.exists():
                raise FileNotFoundError(f"Missing input file: {source}")

            with open(source, "rb") as in_fh:
                shutil.copyfileobj(in_fh, out_fh, length=8 * 1024 * 1024)
            total_bytes += source.stat().st_size

            if idx % 2 == 0:
                dataset_vol.commit()

    if delete_inputs:
        for input_path in input_paths:
            Path(input_path).unlink(missing_ok=True)

    dataset_vol.commit()
    print(f"Wrote combined archive {destination} ({total_bytes / 1e9:.2f} GB)")
    return {
        "output_path": str(destination),
        "bytes_written": total_bytes,
        "num_inputs": len(input_paths),
    }


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=300,
)
def list_existing_archives(archive_root: str) -> list[str]:
    """Return existing ZIP archives under *archive_root* on the volume."""
    dataset_vol.reload()

    root = Path(archive_root)
    if not root.exists():
        return []

    archives = sorted(str(path) for path in root.rglob("*.zip") if path.is_file())
    for archive in archives:
        print(archive)
    return archives


def _shard_prefix(relative_path: Path) -> str:
    """Return a deterministic short subdirectory prefix for *relative_path*."""
    return hashlib.md5(relative_path.as_posix().encode()).hexdigest()[:_SHARD_HEX_CHARS]


def _sharded_member_path(member: Path) -> Path:
    """Insert a hash shard before the file name to avoid giant flat directories."""
    return member.parent / _shard_prefix(member) / member.name


# ---------------------------------------------------------------------------
# Phase 3 - extract ZIP archives
# ---------------------------------------------------------------------------


@app.function(
    volumes={MODAL_DATA_DIR: dataset_vol},
    timeout=28800,
    memory=8192,
    retries=1,
)
def extract_zip_archive(
    archive_path: str,
    extract_dir: str,
    delete_archive: bool = False,
    shard_files: bool = True,
    clear_destination: bool = False,
) -> dict[str, Any]:
    """Extract a ZIP archive from the volume into *extract_dir* safely."""
    import zipfile

    dataset_vol.reload()

    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    destination_root = Path(extract_dir)
    if clear_destination and destination_root.exists():
        print(f"Clearing existing extraction directory: {destination_root}")
        shutil.rmtree(destination_root)
        dataset_vol.commit()

    destination_root.mkdir(parents=True, exist_ok=True)
    root_resolved = destination_root.resolve()

    extracted_files = 0
    extracted_bytes = 0

    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            member = Path(info.filename)
            if info.is_dir():
                continue
            if member.is_absolute() or ".." in member.parts:
                print(f"Skipping suspicious member: {info.filename}")
                continue

            output_member = _sharded_member_path(member) if shard_files else member
            destination = destination_root / output_member
            destination.parent.mkdir(parents=True, exist_ok=True)

            resolved_parent = destination.parent.resolve()
            if os.path.commonpath([str(root_resolved), str(resolved_parent)]) != str(root_resolved):
                print(f"Skipping escaped member: {info.filename}")
                continue

            with zf.open(info) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

            extracted_files += 1
            extracted_bytes += info.file_size

            if extracted_files % 5000 == 0:
                dataset_vol.commit()
                print(f"  ... {extracted_files:,} files extracted")

    if delete_archive:
        archive.unlink(missing_ok=True)

    dataset_vol.commit()
    print(
        f"Extracted {extracted_files:,} files "
        f"({extracted_bytes / 1e9:.2f} GB uncompressed)"
    )
    return {
        "archive_path": str(archive),
        "extract_dir": str(destination_root),
        "num_files": extracted_files,
        "bytes_extracted": extracted_bytes,
    }


# ---------------------------------------------------------------------------
# Helpers for the local entrypoint
# ---------------------------------------------------------------------------


def _build_download_plan(
    file_sizes: dict[str, int],
    download_root: str,
    max_workers: int,
) -> tuple[
    list[tuple[str, int, int, int, str]],
    list[tuple[str, str, int, str]],
]:
    """Compute chunk assignments that fit within *max_workers* containers."""
    total_bytes = sum(file_sizes.values())
    if total_bytes == 0:
        return [], []

    target_chunk_bytes = max(total_bytes // max(max_workers, 1), 50 * 1024 * 1024)

    chunk_args: list[tuple[str, int, int, int, str]] = []
    reassemble_args: list[tuple[str, str, int, str]] = []

    for filename, size in file_sizes.items():
        if size <= 0:
            raise ValueError(f"Refusing to download zero-byte or unknown-size file: {filename}")

        chunk_dir = f"{MODAL_DATA_DIR}/_chunks/{filename}"
        output_path = f"{download_root}/{filename}"

        if size < _CHUNK_THRESHOLD:
            num_chunks = 1
        else:
            num_chunks = max(1, math.ceil(size / target_chunk_bytes))

        actual_chunk_bytes = math.ceil(size / num_chunks)
        for chunk_idx in range(num_chunks):
            start = chunk_idx * actual_chunk_bytes
            end = min(start + actual_chunk_bytes - 1, size - 1)
            chunk_args.append((filename, chunk_idx, start, end, chunk_dir))

        reassemble_args.append((filename, chunk_dir, num_chunks, output_path))

    return chunk_args, reassemble_args


def _group_multipart_zips(repo_files: list[str]) -> dict[str, list[str]]:
    """Return ``{archive_name: [ordered_part_names...]}`` for multipart ZIPs."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for repo_file in repo_files:
        match = _MULTIPART_ZIP_RE.match(repo_file)
        if match is None:
            continue
        grouped[match.group("base")].append((int(match.group("part")), repo_file))

    archives: dict[str, list[str]] = {}
    for archive_name, parts in grouped.items():
        ordered_parts = sorted(parts)
        expected = list(range(1, len(ordered_parts) + 1))
        actual = [part_num for part_num, _ in ordered_parts]
        if actual != expected:
            raise ValueError(
                f"Missing multipart ZIP segment(s) for {archive_name}: "
                f"expected {expected}, got {actual}"
            )
        archives[archive_name] = [repo_file for _, repo_file in ordered_parts]

    return archives


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    output_subdir: str = "tablebank",
    max_workers: int = 100,
    include_metadata: bool = True,
    extract_archives: bool = True,
    keep_split_parts: bool = True,
    keep_archives: bool = True,
    shard_extracted_files: bool = True,
    clear_existing_extract_dir: bool = False,
    extract_only: bool = False,
):
    """Download TableBank into the tablebank-vol Modal volume."""
    raw_root = f"{MODAL_DATA_DIR}/{output_subdir}/raw"
    archive_root = f"{MODAL_DATA_DIR}/{output_subdir}/archives"
    extract_root = f"{MODAL_DATA_DIR}/{output_subdir}/extracted"

    print(f"{'Dataset':>16}: {DATASET_ID}")
    print(f"{'Volume':>16}: tablebank-vol")
    print(f"{'Volume path':>16}: /data/{output_subdir}")
    print(f"{'Max workers':>16}: {max_workers}")
    print(f"{'Metadata files':>16}: {'yes' if include_metadata else 'no'}")
    print(f"{'Extract zips':>16}: {'yes' if extract_archives else 'no'}")
    print(f"{'Keep parts':>16}: {'yes' if keep_split_parts else 'no'}")
    print(f"{'Keep archives':>16}: {'yes' if keep_archives else 'no'}")
    print(f"{'Shard extract':>16}: {'yes' if shard_extracted_files else 'no'}")
    print(f"{'Clear extract':>16}: {'yes' if clear_existing_extract_dir else 'no'}")
    print(f"{'Extract only':>16}: {'yes' if extract_only else 'no'}")
    print()

    combined_archives: list[dict[str, Any]]
    if extract_only:
        print("Extract-only mode: reusing existing combined archives ...")
        archive_paths = list_existing_archives.remote(archive_root)
        if not archive_paths:
            raise RuntimeError(
                f"No existing archives found under {archive_root}. "
                "Run without --extract-only first."
            )
        combined_archives = [{"output_path": archive_path} for archive_path in archive_paths]
        for archive_path in archive_paths:
            print(f"  {archive_path}")
        print()
    else:
        print("Phase 0: listing dataset files ...")
        repo_files = list_dataset_files.remote(include_metadata)
        if not repo_files:
            raise RuntimeError("No dataset files found")
        print(f"  Found {len(repo_files)} files\n")

        print("Phase 1a: probing file sizes ...")
        file_sizes = probe_files.remote(repo_files)
        total_gb = sum(file_sizes.values()) / 1e9
        print(f"  Total: {total_gb:.1f} GB across {len(file_sizes)} files\n")

        chunk_args, reassemble_args = _build_download_plan(file_sizes, raw_root, max_workers)

        print(
            f"Phase 1b: downloading via {len(chunk_args)} parallel chunk "
            f"containers ..."
        )
        list(download_chunk.starmap(chunk_args))
        print("  All chunks downloaded.\n")

        print(f"Phase 1c: reassembling {len(reassemble_args)} files ...")
        reassembled = list(reassemble_to_file.starmap(reassemble_args))
        raw_path_by_repo_file = {
            item["repo_file"]: item["output_path"]
            for item in reassembled
        }
        for item in reassembled:
            print(f"  {item['repo_file']} -> {item['output_path']}")
        print()

        multipart_zips = _group_multipart_zips(list(raw_path_by_repo_file))
        if not multipart_zips:
            print("No multipart ZIP archives found.")
            print(f"\nDone! Raw files are in Modal volume 'tablebank-vol' at /data/{output_subdir}")
            return

        print(f"Phase 2: concatenating {len(multipart_zips)} multipart ZIP archive(s) ...")
        concat_args: list[tuple[list[str], str, bool]] = []
        for archive_name, part_files in multipart_zips.items():
            part_paths = [raw_path_by_repo_file[part_file] for part_file in part_files]
            archive_path = f"{archive_root}/{archive_name}"
            concat_args.append((part_paths, archive_path, not keep_split_parts))

        combined_archives = list(concatenate_files.starmap(concat_args))
        for archive in combined_archives:
            print(f"  {archive['output_path']} ({archive['bytes_written'] / 1e9:.2f} GB)")
        print()

    if extract_archives:
        print(f"Phase 3: extracting {len(combined_archives)} ZIP archive(s) ...")
        extract_args: list[tuple[str, str, bool, bool, bool]] = []
        for archive in combined_archives:
            archive_path = archive["output_path"]
            archive_name = Path(archive_path).stem
            destination = f"{extract_root}/{archive_name}"
            extract_args.append((
                archive_path,
                destination,
                not keep_archives,
                shard_extracted_files,
                clear_existing_extract_dir,
            ))

        extracted = list(extract_zip_archive.starmap(extract_args))
        for item in extracted:
            print(f"  {item['archive_path']} -> {item['extract_dir']}")
            print(f"    {item['num_files']:,} files")
        print()

    print("Done!")
    print(f"  Raw files: /data/{output_subdir}/raw")
    print(f"  Archives:  /data/{output_subdir}/archives")
    if extract_archives:
        print(f"  Extracted: /data/{output_subdir}/extracted")
