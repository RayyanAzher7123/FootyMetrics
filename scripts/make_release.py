"""Build a portfolio zip without local environments or bulk source downloads.

The archive keeps the joined CSV tables and the trained joblib files so the
demo can run. It leaves out node_modules, virtualenvs, and the raw FPL,
football-data, and Transfermarkt downloads those tables were built from.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "releases" / "FootyMetrics.zip"

SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vite",
    "__pycache__",
    "dist",
    "node_modules",
    "releases",
    "venv",
}
SKIP_PREFIXES = (
    "data/raw/football_data/",
    "data/raw/fpl/",
    "data/raw/transfermarkt/",
)
SKIP_SUFFIXES = {".log", ".pyc"}


def should_pack(relative: Path) -> bool:
    parts = relative.parts
    if any(part in SKIP_DIR_NAMES for part in parts):
        return False
    posix = relative.as_posix()
    if posix.startswith(SKIP_PREFIXES):
        return False
    return relative.suffix.lower() not in SKIP_SUFFIXES


def release_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and should_pack(path.relative_to(root))
    ]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def write_release(root: Path = ROOT, destination: Path = DEFAULT_DESTINATION) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in release_files(root):
            archive.write(path, path.relative_to(root).as_posix())
    return destination


def main() -> None:
    destination = write_release()
    with zipfile.ZipFile(destination) as archive:
        infos = archive.infolist()
    uncompressed = sum(info.file_size for info in infos)
    print(f"Wrote {destination}")
    print(f"{len(infos)} files, {uncompressed / 1_000_000:.1f} MB uncompressed")


if __name__ == "__main__":
    main()
