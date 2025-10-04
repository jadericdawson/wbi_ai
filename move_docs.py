#!/usr/bin/env python3
"""
move_docs.py
Recursively find all .docx and .pdf files under a source directory and move (or copy)
them into a single destination directory.

Features:
- Safe handling of name collisions (appends a numeric suffix or content hash).
- Optional dry-run to preview actions.
- Optional copy mode (default is move).
- Preserves file timestamps/metadata when copying.
- Skips the destination folder if it’s inside the source tree.
- Optional exclusion of hidden files/folders.
- Detailed logging to console and optional log file.

Usage examples:
  # Move all docs to a default "_collected_docs" folder inside the source
  python3 move_docs.py \
      --src "/home/jadericdawson/Documents/WBI/WBI-IL-2024-09-003 F-35 Manpower Model Improvement Initiative/"

  # Move to a specific destination outside the source
  python3 move_docs.py \
      --src "/home/.../F-35 Manpower Model Improvement Initiative/" \
      --dst "/home/jadericdawson/Documents/WBI/_collected_docs"

  # Copy instead of move, and do a dry run first
  python3 move_docs.py --src "/home/.../F-35..." --copy --dry-run

  # Log to a file
  python3 move_docs.py --src "/home/.../F-35..." --log "/home/jadericdawson/move_docs.log"
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Tuple

DOC_EXTS = {".pdf", ".docx"}

def is_hidden(path: Path) -> bool:
    parts = path.resolve().parts
    return any(p.startswith('.') for p in parts)

def iter_files(root: Path, include_hidden: bool) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        # Optionally skip hidden directories
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for name in filenames:
            p = dp / name
            if not include_hidden and is_hidden(p):
                continue
            if p.suffix.lower() in DOC_EXTS:
                yield p

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def file_sha1(p: Path, block_size: int = 65536) -> str:
    h = hashlib.sha1()
    with p.open('rb') as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:8]  # short hash for suffix

def unique_destination(dst_dir: Path, src_file: Path) -> Path:
    """
    Return a non-conflicting destination path. If a file with the same name exists,
    append '-1', '-2', ... If contents are identical (hash match), we still suffix
    to avoid overwriting, but try to use a short content hash first.
    """
    base = src_file.stem
    ext = src_file.suffix
    candidate = dst_dir / f"{base}{ext}"
    if not candidate.exists():
        return candidate

    # Try content-hash suffix first
    try:
        h = file_sha1(src_file)
        candidate = dst_dir / f"{base}-{h}{ext}"
        if not candidate.exists():
            return candidate
    except Exception:
        # If hashing fails, fall through to numeric suffixing
        pass

    # Numeric suffix fallback
    i = 1
    while True:
        candidate = dst_dir / f"{base}-{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1

def safe_move_or_copy(src: Path, dst: Path, do_copy: bool, dry_run: bool) -> Tuple[bool, str]:
    """
    Move or copy src -> dst. Returns (success, message).
    """
    action = "COPY" if do_copy else "MOVE"
    if dry_run:
        return True, f"[DRY RUN] {action}: {src} -> {dst}"

    try:
        ensure_dir(dst.parent)
        if do_copy:
            shutil.copy2(src, dst)  # preserves metadata
        else:
            # If moving across filesystems, fallback to copy+remove
            try:
                src.rename(dst)
            except OSError:
                shutil.copy2(src, dst)
                src.unlink()
        return True, f"{action}: {src} -> {dst}"
    except Exception as e:
        return False, f"ERROR {action} {src} -> {dst}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Collect .pdf and .docx files into one folder.")
    parser.add_argument("--src", required=True, help="Source directory to scan recursively.")
    parser.add_argument("--dst", help="Destination directory (default: <src>/_collected_docs)")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move (default is move).")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without changing files.")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files/folders.")
    parser.add_argument("--log", help="Optional log file path.")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        print(f"Source directory does not exist or is not a directory: {src}", file=sys.stderr)
        sys.exit(1)

    dst = Path(args.dst).expanduser().resolve() if args.dst else (src.parent / "_collected_docs")
    # If destination is inside source, ensure we skip that subtree during walks
    dst_in_src = str(dst).startswith(str(src))

    if not args.dry_run:
        ensure_dir(dst)

    # Open log file if requested
    log_fh = None
    if args.log:
        log_path = Path(args.log).expanduser().resolve()
        ensure_dir(log_path.parent)
        log_fh = log_path.open("a", encoding="utf-8")

    def log(msg: str):
        print(msg)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    log("=== move_docs.py started ===")
    log(f"Source:        {src}")
    log(f"Destination:   {dst}")
    log(f"Mode:          {'COPY' if args.copy else 'MOVE'}")
    log(f"Dry run:       {args.dry_run}")
    log(f"Include hidden:{args.include_hidden}")
    if args.log:
        log(f"Log file:      {Path(args.log).resolve()}")

    count = 0
    skipped = 0
    for f in iter_files(src, include_hidden=args.include_hidden):
        # Skip anything already in the destination subtree
        if dst_in_src and str(f.resolve()).startswith(str(dst)):
            skipped += 1
            continue

        target = unique_destination(dst, f)
        ok, message = safe_move_or_copy(f, target, do_copy=args.copy, dry_run=args.dry_run)
        log(message)
        if ok:
            count += 1

    log(f"Completed. Files processed: {count}. Skipped (in dst): {skipped}.")
    log("=== move_docs.py finished ===")

    if log_fh:
        log_fh.close()

if __name__ == "__main__":
    main()
