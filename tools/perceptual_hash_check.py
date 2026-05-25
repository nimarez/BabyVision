#!/usr/bin/env python3
"""Check generated BabyVision images against dataset samples with image hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a convenience only.
    tqdm = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageBlob:
    ref: str
    source: str
    name: str
    data: bytes


@dataclass(frozen=True)
class ImageHashes:
    ref: str
    source: str
    name: str
    width: int
    height: int
    mode: str
    sha256: str
    pixel_sha256: str
    ahash: str
    dhash: str
    phash: str


@dataclass(frozen=True)
class MatchRow:
    generated: str
    dataset: str
    match_type: str
    exact_byte: bool
    exact_pixel: bool
    ahash_distance: int
    dhash_distance: int
    phash_distance: int
    dataset_source: str
    generated_source: str


def is_image_name(name: str) -> bool:
    path = Path(name)
    return (
        path.suffix.lower() in IMAGE_SUFFIXES
        and "__MACOSX/" not in name
        and not path.name.startswith("._")
        and path.name != ".DS_Store"
    )


def iter_image_blobs(source: Path) -> Iterable[ImageBlob]:
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and is_image_name(path.name):
                yield ImageBlob(
                    ref=str(path),
                    source=str(source),
                    name=str(path.relative_to(source)),
                    data=path.read_bytes(),
                )
        return

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or not is_image_name(info.filename):
                    continue
                yield ImageBlob(
                    ref=f"zip://{source}!{info.filename}",
                    source=str(source),
                    name=info.filename,
                    data=archive.read(info),
                )
        return

    if source.is_file() and is_image_name(source.name):
        yield ImageBlob(
            ref=str(source),
            source=str(source.parent),
            name=source.name,
            data=source.read_bytes(),
        )
        return

    raise ValueError(f"Unsupported source: {source}")


def prepared_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def grayscale_for_hash(image: Image.Image) -> Image.Image:
    return prepared_rgb(image).convert("L")


def normalized_pixel_sha256(image: Image.Image) -> str:
    normalized = prepared_rgb(image)
    digest = hashlib.sha256()
    digest.update(normalized.mode.encode("ascii"))
    digest.update(str(normalized.size).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1).astype(bool):
        value = (value << 1) | int(bit)
    width = (bits.size + 3) // 4
    return f"{value:0{width}x}"


def average_hash(image: Image.Image, hash_size: int) -> str:
    small = grayscale_for_hash(image).resize(
        (hash_size, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(small, dtype=np.float32)
    return bits_to_hex(pixels > pixels.mean())


def difference_hash(image: Image.Image, hash_size: int) -> str:
    small = grayscale_for_hash(image).resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(small, dtype=np.float32)
    return bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def dct_matrix(size: int) -> np.ndarray:
    indices = np.arange(size)
    matrix = np.empty((size, size), dtype=np.float32)
    for frequency in range(size):
        scale = np.sqrt(1 / size) if frequency == 0 else np.sqrt(2 / size)
        matrix[frequency, :] = scale * np.cos(
            ((2 * indices + 1) * frequency * np.pi) / (2 * size)
        )
    return matrix


def perceptual_hash(image: Image.Image, hash_size: int, highfreq_factor: int) -> str:
    image_size = hash_size * highfreq_factor
    small = grayscale_for_hash(image).resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(small, dtype=np.float32)
    transform = dct_matrix(image_size)
    dct = transform @ pixels @ transform.T
    lowfreq = dct[:hash_size, :hash_size].copy()
    median = np.median(lowfreq.reshape(-1)[1:])
    bits = lowfreq > median
    bits[0, 0] = False
    return bits_to_hex(bits)


def hamming_distance(left_hex: str, right_hex: str) -> int:
    return (int(left_hex, 16) ^ int(right_hex, 16)).bit_count()


def hash_blob(
    blob: ImageBlob,
    *,
    hash_size: int,
    highfreq_factor: int,
) -> ImageHashes | None:
    try:
        with Image.open(io.BytesIO(blob.data)) as image:
            image.load()
            rgb = prepared_rgb(image)
            return ImageHashes(
                ref=blob.ref,
                source=blob.source,
                name=blob.name,
                width=rgb.width,
                height=rgb.height,
                mode=rgb.mode,
                sha256=hashlib.sha256(blob.data).hexdigest(),
                pixel_sha256=normalized_pixel_sha256(rgb),
                ahash=average_hash(rgb, hash_size),
                dhash=difference_hash(rgb, hash_size),
                phash=perceptual_hash(rgb, hash_size, highfreq_factor),
            )
    except (OSError, UnidentifiedImageError) as exc:
        print(f"warning: could not read image {blob.ref}: {exc}", file=sys.stderr)
        return None


def load_hashes(
    sources: list[Path],
    *,
    hash_size: int,
    highfreq_factor: int,
    label: str,
    show_progress: bool,
) -> list[ImageHashes]:
    blobs = [blob for source in sources for blob in iter_image_blobs(source)]
    iterator = blobs
    if show_progress and tqdm is not None:
        iterator = tqdm(blobs, desc=f"hash {label}", unit="image")

    hashes: list[ImageHashes] = []
    for blob in iterator:
        hashed = hash_blob(blob, hash_size=hash_size, highfreq_factor=highfreq_factor)
        if hashed is not None:
            hashes.append(hashed)
    return hashes


def compare_images(
    generated: list[ImageHashes],
    dataset: list[ImageHashes],
    *,
    max_distance: int,
    top_k: int,
) -> tuple[list[dict], list[MatchRow]]:
    results: list[dict] = []
    report_rows: list[MatchRow] = []

    for generated_item in generated:
        matches: list[MatchRow] = []
        nearest: list[MatchRow] = []

        for dataset_item in dataset:
            ahash_distance = hamming_distance(generated_item.ahash, dataset_item.ahash)
            dhash_distance = hamming_distance(generated_item.dhash, dataset_item.dhash)
            phash_distance = hamming_distance(generated_item.phash, dataset_item.phash)
            exact_byte = generated_item.sha256 == dataset_item.sha256
            exact_pixel = generated_item.pixel_sha256 == dataset_item.pixel_sha256

            row = MatchRow(
                generated=generated_item.ref,
                dataset=dataset_item.ref,
                match_type="nearest",
                exact_byte=exact_byte,
                exact_pixel=exact_pixel,
                ahash_distance=ahash_distance,
                dhash_distance=dhash_distance,
                phash_distance=phash_distance,
                dataset_source=dataset_item.source,
                generated_source=generated_item.source,
            )
            nearest.append(row)

            if exact_byte:
                match_type = "exact_byte"
            elif exact_pixel:
                match_type = "exact_pixel"
            elif (
                ahash_distance <= max_distance
                or dhash_distance <= max_distance
                or phash_distance <= max_distance
            ):
                match_type = "perceptual"
            else:
                continue

            matches.append(
                MatchRow(
                    generated=row.generated,
                    dataset=row.dataset,
                    match_type=match_type,
                    exact_byte=row.exact_byte,
                    exact_pixel=row.exact_pixel,
                    ahash_distance=row.ahash_distance,
                    dhash_distance=row.dhash_distance,
                    phash_distance=row.phash_distance,
                    dataset_source=row.dataset_source,
                    generated_source=row.generated_source,
                )
            )

        nearest = sorted(
            nearest,
            key=lambda row: (
                row.phash_distance,
                row.dhash_distance,
                row.ahash_distance,
                row.dataset,
            ),
        )[:top_k]
        report_rows.extend(matches)
        report_rows.extend(nearest)
        results.append(
            {
                "generated": asdict(generated_item),
                "matches": [asdict(row) for row in matches],
                "nearest": [asdict(row) for row in nearest],
            }
        )

    return results, report_rows


def fail_count(rows: list[MatchRow], fail_on: str) -> int:
    if fail_on == "none":
        return 0
    if fail_on == "byte":
        return sum(row.exact_byte for row in rows if row.match_type != "nearest")
    if fail_on == "pixel":
        return sum(
            row.exact_byte or row.exact_pixel
            for row in rows
            if row.match_type != "nearest"
        )
    if fail_on == "perceptual":
        return sum(1 for row in rows if row.match_type != "nearest")
    raise ValueError(f"Unsupported fail mode: {fail_on}")


def write_csv(path: Path, rows: list[MatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MatchRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare generated images against dataset samples using exact hashes, "
            "normalized pixel hashes, and perceptual hashes."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        required=True,
        help="Dataset image directory, image file, or zip archive. Can be repeated.",
    )
    parser.add_argument(
        "--generated",
        action="append",
        type=Path,
        required=True,
        help="Generated image directory, image file, or zip archive. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_outputs/perceptual_hash_report.json"),
        help="JSON report path.",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV report path.")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=6,
        help="Maximum Hamming distance for perceptual near-match flags.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest dataset neighbors to keep for each generated image.",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Perceptual hash size. 8 gives 64-bit hashes.",
    )
    parser.add_argument(
        "--highfreq-factor",
        type=int,
        default=4,
        help="pHash resize multiplier before DCT.",
    )
    parser.add_argument(
        "--fail-on",
        choices=["none", "byte", "pixel", "perceptual"],
        default="pixel",
        help=(
            "Exit nonzero on byte-exact matches, normalized pixel-exact matches, "
            "or perceptual matches. Defaults to pixel."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_hashes = load_hashes(
        args.dataset,
        hash_size=args.hash_size,
        highfreq_factor=args.highfreq_factor,
        label="dataset",
        show_progress=not args.no_progress,
    )
    generated_hashes = load_hashes(
        args.generated,
        hash_size=args.hash_size,
        highfreq_factor=args.highfreq_factor,
        label="generated",
        show_progress=not args.no_progress,
    )

    results, rows = compare_images(
        generated_hashes,
        dataset_hashes,
        max_distance=args.max_distance,
        top_k=args.top_k,
    )
    match_rows = [row for row in rows if row.match_type != "nearest"]
    summary = {
        "dataset_sources": [str(path) for path in args.dataset],
        "generated_sources": [str(path) for path in args.generated],
        "dataset_image_count": len(dataset_hashes),
        "generated_image_count": len(generated_hashes),
        "hash_size": args.hash_size,
        "max_distance": args.max_distance,
        "exact_byte_match_count": sum(row.exact_byte for row in match_rows),
        "exact_pixel_match_count": sum(row.exact_pixel for row in match_rows),
        "perceptual_match_count": sum(
            row.match_type == "perceptual" for row in match_rows
        ),
        "match_count": len(match_rows),
        "fail_on": args.fail_on,
    }
    report = {"summary": summary, "generated": results}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        write_csv(args.csv, rows)

    print(json.dumps(summary, indent=2))
    failures = fail_count(match_rows, args.fail_on)
    if failures:
        print(f"failed: {failures} {args.fail_on} match(es) found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
