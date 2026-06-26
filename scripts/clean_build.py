#!/usr/bin/env python3
"""Remove local build, test, and extension artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DIRECTORIES = [
    ROOT / "build",
    ROOT / "dist",
    ROOT / "_skbuild",
    ROOT / ".pytest_cache",
]

GLOBS = [
    "*.egg-info",
    "*.egg",
    "vasapy/_vasapy*.so",
]

PYTHON_ARTIFACT_ROOTS = [
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "vasapy",
]


def remove_path(path: Path, dry_run: bool) -> bool:
    if not path.exists() and not path.is_symlink():
        return False

    relpath = path.relative_to(ROOT)
    if dry_run:
        print(f"would remove {relpath}")
        return True

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"removed {relpath}")
    return True


def iter_targets() -> list[Path]:
    targets = set(DIRECTORIES)
    for pattern in GLOBS:
        targets.update(ROOT.glob(pattern))
    for artifact_root in PYTHON_ARTIFACT_ROOTS:
        if artifact_root.exists():
            targets.update(artifact_root.rglob("__pycache__"))
            targets.update(artifact_root.rglob("*.pyc"))
            targets.update(artifact_root.rglob("*.pyo"))

    existing = sorted(
        (path for path in targets if path.exists() or path.is_symlink()),
        key=lambda path: (len(path.parts), str(path)),
    )
    pruned: list[Path] = []
    for path in existing:
        if any(parent in pruned for parent in path.parents):
            continue
        pruned.append(path)
    return pruned


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove local build, test, and extension artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be removed without deleting anything",
    )
    args = parser.parse_args()

    removed = sum(remove_path(path, args.dry_run) for path in iter_targets())
    if removed == 0:
        print("nothing to clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
