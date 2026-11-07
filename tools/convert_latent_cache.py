"""Convert a folder of latent caches in place to another storage dtype (fp32 / fp16 / bf16).

The caches written by `--cache_latents_to_disk` are npz files holding the latents of every bucket resolution an image is
trained at (`latents_<H>x<W>`, `latents_flipped_<H>x<W>`) plus the original size, the crop and optionally the alpha mask.
`--latent_cache_dtype` chooses the dtype of the latent arrays; this tool converts files written with another setting
without touching the dataset, the images or the training run that will consume them.

Usage:

    python tools/convert_latent_cache.py --dir /home/wd/data/img/pu_ht --to bf16 --dry-run
    python tools/convert_latent_cache.py --dir /home/wd/data/img/pu_ht --to bf16
    # caches next to the images of nested folders (the dataset's `recursive`):
    python tools/convert_latent_cache.py --dir /home/wd/data/img --recursive --to bf16

Each file is rewritten through a temporary file in the same directory and then renamed, so an interrupted run leaves every
cache either fully converted or untouched. Files that already hold the target dtype (or hold no latents) are skipped, so
the tool is idempotent. Set the training config's `latent_cache_dtype` to the same value afterwards, otherwise the trainer
re-caches (it treats a dtype mismatch as a stale cache).
"""

import argparse
import glob
import os
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from library import caching  # noqa: E402


def latent_keys(npz) -> List[str]:
    """The latent arrays of a cache file: `latents`, `latents_<H>x<W>`, and the flipped ones (not the dtype marker)."""
    return [
        key
        for key in npz.files
        if key != caching.STORAGE_DTYPE_KEY
        and (key == "latents" or key.startswith("latents_") or key.startswith("latents_flipped"))
    ]


def stored_dtype(npz) -> Optional[str]:
    """The dtype a cache file holds, from its marker (absent = fp32, the format before the marker existed)."""
    if caching.STORAGE_DTYPE_KEY in npz:
        return str(npz[caching.STORAGE_DTYPE_KEY])
    keys = latent_keys(npz)
    if not keys:
        return None
    array = npz[keys[0]]
    if array.dtype == np.uint16:
        return "bf16"
    if array.dtype == np.float16:
        return "fp16"
    return "fp32"


def convert_file(path: str, target: str, dry_run: bool = False) -> Optional[str]:
    """Convert one npz in place. Returns a status word: converted, already, skipped (no latents), or dry-run text."""
    with np.load(path) as npz:
        source = stored_dtype(npz)
        keys = latent_keys(npz)
        if source is None or not keys:
            return "skipped"
        if source == target:
            return "already"
        kwargs = {key: npz[key] for key in npz.files}

    for key in keys:
        # through float32, so an fp16 <-> bf16 conversion does not round twice
        kwargs[key] = caching.latent_to_storage(caching.latent_from_storage(kwargs[key]).float(), target)
    kwargs[caching.STORAGE_DTYPE_KEY] = np.array(target)

    if dry_run:
        print(f"  would convert {os.path.basename(path)}: {source} -> {target}")
        return "converted"

    directory = os.path.dirname(path) or "."
    temporary = os.path.join(directory, "." + os.path.basename(path) + ".tmp")
    # np.savez appends ".npz" to a path it does not recognise, so hand it a file object instead
    with open(temporary, "wb") as handle:
        np.savez(handle, **kwargs)
    os.replace(temporary, path)  # atomic within the same filesystem
    return "converted"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="directory that contains the latent caches")
    parser.add_argument("--to", required=True, choices=list(caching.LATENT_STORAGE_DTYPES), help="target dtype")
    parser.add_argument("--pattern", default="*.npz", help="glob of the cache files (default: *.npz)")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="also search the subdirectories of --dir (the same meaning as `recursive` in the dataset config)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args()

    pattern = os.path.join(args.dir, "**", args.pattern) if args.recursive else os.path.join(args.dir, args.pattern)
    files = sorted(glob.glob(pattern, recursive=args.recursive))
    print(f"searching {'recursively ' if args.recursive else ''}{args.dir}")
    if not files:
        print(f"no files matching {args.pattern} under {args.dir}")
        return 1

    counts = {"converted": 0, "already": 0, "skipped": 0}
    before = after = 0
    for path in files:
        try:
            before += os.path.getsize(path)
            status = convert_file(path, args.to, args.dry_run)
            after += os.path.getsize(path) if not args.dry_run else before
        except Exception as e:  # keep going: one broken file must not abort the conversion
            print(f"  failed: {path}: {type(e).__name__}: {e}")
            counts["skipped"] += 1
            continue
        if status in counts:
            counts[status] += 1

    verb = "would convert" if args.dry_run else "converted"
    print(
        f"{len(files)} files: {counts['converted']} {verb}, {counts['already']} already {args.to}, {counts['skipped']} skipped"
    )
    if not args.dry_run and counts["converted"]:
        print(f"size {before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB")
        print(f"set latent_cache_dtype = '{args.to}' in the training config so the trainer accepts these caches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
