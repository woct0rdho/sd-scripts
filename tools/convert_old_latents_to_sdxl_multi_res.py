import argparse
import collections
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np


SOURCE_NAME_RE = re.compile(r"^(?P<base>.*)_(?P<w>\d+)x(?P<h>\d+)\.npz$")
MERGED_KEY_BASES = ("latents", "original_size", "crop_ltrb", "latents_flipped", "alpha_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert old per-scale latent .npz files into merged multi-resolution "
            "SDXL latent .npz files with suffixed keys."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Directory containing old latent files named like '<base>_<w>x<h>.npz'",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Directory to write merged latent files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. By default, existing outputs are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned output files without writing them.",
    )
    return parser.parse_args()


def collect_groups(src_dir: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    for src_path in glob.glob(os.path.join(src_dir, "*.npz")):
        name = os.path.basename(src_path)
        m = SOURCE_NAME_RE.match(name)
        if m is None:
            raise ValueError(f"Unexpected source filename format: {name}")
        groups[m.group("base")].append(src_path)
    return groups


def convert_group(base: str, paths: List[str]) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    merged: Dict[str, np.ndarray] = {}
    original_size: Tuple[int, int] | None = None

    for path in sorted(paths):
        with np.load(path) as npz:
            if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:
                raise ValueError(f"Missing required keys in {path}; required: latents, original_size, crop_ltrb")

            this_original_size = tuple(int(v) for v in npz["original_size"].tolist())
            if original_size is None:
                original_size = this_original_size
            elif original_size != this_original_size:
                raise ValueError(
                    f"Inconsistent original_size for '{base}': {original_size} vs {this_original_size} ({path})"
                )

            latents_shape = npz["latents"].shape
            if len(latents_shape) < 2:
                raise ValueError(f"Invalid latents shape in {path}: {latents_shape}")
            lat_h = int(latents_shape[-2])
            lat_w = int(latents_shape[-1])
            suffix = f"_{lat_h}x{lat_w}"

            for key_base in MERGED_KEY_BASES:
                if key_base not in npz:
                    continue
                merged_key = key_base + suffix
                if merged_key in merged:
                    raise ValueError(f"Duplicate merged key '{merged_key}' for base '{base}'")
                merged[merged_key] = npz[key_base]

    if original_size is None:
        raise ValueError(f"No input files found for base '{base}'")

    return merged, original_size


def main() -> None:
    args = parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    groups = collect_groups(src_dir)
    if not groups:
        print(f"No source .npz files found in {src_dir}")
        return

    written = 0
    skipped = 0

    for base in sorted(groups.keys()):
        merged, original_size = convert_group(base, groups[base])
        out_name = f"{base}_{original_size[0]:04}x{original_size[1]:04}_sdxl.npz"
        out_path = os.path.join(dst_dir, out_name)

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            print(f"[skip] {out_path}")
            continue

        if args.dry_run:
            print(f"[plan] {out_path}")
            written += 1
            continue

        np.savez(out_path, **merged)
        written += 1
        print(f"[ok] {out_path}")

    action_word = "planned" if args.dry_run else "written"
    print(
        f"Done. groups={len(groups)} {action_word}={written} skipped_existing={skipped} "
        f"src={src_dir} dst={dst_dir}"
    )


if __name__ == "__main__":
    main()
