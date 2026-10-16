"""Download PlantVillage color images into the project's expected folder layout.

Source: https://huggingface.co/datasets/mohanty/PlantVillage
The full source archive is large (about 2 GB). The archive itself and extracted
images are kept outside Git history; do not commit the dataset to this repo.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "plantvillage" / "color"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
)
EXPECTED_PREFIX = "raw/color/"


def download_archive(destination: Path) -> None:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "plant-doctor-ai-dataset-preparer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 // total
                    print(
                        f"\rDownloading: {downloaded:,}/{total:,} bytes ({percent}%)",
                        end="",
                        flush=True,
                    )
    print()


def prepare(destination: Path, replace: bool = False) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        if not replace:
            raise FileExistsError(
                f"{destination} is not empty. Move/delete it or rerun with --replace."
            )
        shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

    archive_path = destination.parent / "plantvillage-data.zip"
    try:
        print(f"Source: {ARCHIVE_URL}")
        print("Downloading the full archive (approximately 2 GB); this may take a while.")
        download_archive(archive_path)

        extracted = 0
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                item for item in archive.infolist()
                if not item.is_dir() and item.filename.startswith(EXPECTED_PREFIX)
            ]
            if not members:
                raise RuntimeError(
                    f"No files under {EXPECTED_PREFIX!r} found in source archive."
                )

            for index, item in enumerate(members, start=1):
                relative = Path(item.filename[len(EXPECTED_PREFIX):])
                if not relative.parts or ".." in relative.parts:
                    continue
                # Keep only image files stored inside class folders.
                if relative.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
                if index % 500 == 0:
                    print(f"Organized {extracted:,} images...")

        class_dirs = [path for path in destination.iterdir() if path.is_dir()]
        if not class_dirs or extracted == 0:
            raise RuntimeError("Extraction completed but no class folders/images were found.")
        print(f"Complete: {extracted:,} images across {len(class_dirs)} class folders.")
        print(f"Dataset directory: {destination}")
        print("Expected structure: data/plantvillage/color/<class-name>/<image-file>")
        print("Dataset source license is listed as CC BY-SA 3.0; retain attribution.")
        return extracted
    finally:
        if archive_path.exists():
            archive_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing data/plantvillage/color directory",
    )
    args = parser.parse_args()
    try:
        prepare(DATA_DIR, replace=args.replace)
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
