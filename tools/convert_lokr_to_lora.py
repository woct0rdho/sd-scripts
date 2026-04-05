#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from library import train_util


LOKR_SUFFIXES = {
    "lokr_w1",
    "lokr_w2",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
}

SHARED_SUFFIXES = {
    "alpha",
    "dora_scale",
}

SAVE_PRECISION_MAP = {
    "float": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

MIN_SV = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a LoKr safetensors adapter into a LoRA safetensors adapter "
            "using a rank-limited SVD approximation."
        )
    )
    parser.add_argument("input", type=Path, help="Input LoKr .safetensors file")
    parser.add_argument("output", type=Path, help="Output LoRA .safetensors file")
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Target LoRA rank",
    )
    parser.add_argument(
        "--alpha",
        default="rank",
        help=(
            "Output LoRA alpha. Use 'rank' (default) to make alpha equal the saved "
            "per-layer rank, 'none' to omit alpha tensors, or a numeric value."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for SVD work: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=("float", "fp16", "bf16"),
        help="precision in saving; if omitted, preserve source dtype",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file",
    )
    parser.add_argument(
        "--dynamic_method",
        type=str,
        default=None,
        choices=("sv_ratio", "sv_fro", "sv_cumulative"),
        help="Specify dynamic resizing method, --rank is used as a hard limit for max rank",
    )
    parser.add_argument("--dynamic_param", type=float, default=None, help="Specify target for dynamic reduction")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display verbose conversion statistics",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def split_key_suffix(key: str) -> tuple[str, Optional[str]]:
    for suffix in sorted(LOKR_SUFFIXES | SHARED_SUFFIXES, key=len, reverse=True):
        marker = f".{suffix}"
        if key.endswith(marker):
            return key[: -len(marker)], suffix
    return key, None


def build_lokr_prefixes(keys: list[str]) -> set[str]:
    prefixes: set[str] = set()
    for key in keys:
        prefix, suffix = split_key_suffix(key)
        if suffix in LOKR_SUFFIXES:
            prefixes.add(prefix)
    return prefixes


def output_dtype_for(
    prefix: str,
    reader: safe_open,
    key_set: set[str],
    save_precision_arg: Optional[str],
) -> torch.dtype:
    if save_precision_arg is not None:
        return SAVE_PRECISION_MAP[save_precision_arg]

    for suffix in ("lokr_w2", "lokr_w2_a", "lokr_t2", "lokr_w1", "lokr_w1_a"):
        key = f"{prefix}.{suffix}"
        if key in key_set:
            return reader.get_tensor(key).dtype
    return torch.float32


def format_metadata_scalar(value: Optional[float]) -> str:
    if value is None:
        return "None"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value)


def unify_metadata_scalars(values: list[str], fallback: str) -> str:
    if not values:
        return fallback
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return unique_values[0]
    return "Dynamic"


def build_network_args(metadata: Optional[dict[str, str]], args: argparse.Namespace) -> dict[str, str]:
    existing = None if metadata is None else metadata.get("ss_network_args")
    if existing:
        try:
            parsed = json.loads(existing)
        except json.JSONDecodeError:
            parsed = None
    else:
        parsed = None

    if not isinstance(parsed, dict):
        parsed = {}

    parsed["algo"] = "lora"
    parsed["converted_from"] = "lokr"
    parsed["convert_rank"] = str(args.rank)
    parsed["convert_alpha"] = str(args.alpha)
    if args.dynamic_method is not None:
        parsed["dynamic_method"] = args.dynamic_method
        parsed["dynamic_param"] = str(args.dynamic_param)
    else:
        parsed.pop("dynamic_method", None)
        parsed.pop("dynamic_param", None)
    parsed.pop("factor", None)
    return {k: str(v) for k, v in parsed.items()}


def metadata_for_output(
    metadata: Optional[dict[str, str]],
    args: argparse.Namespace,
    output_ranks: list[str],
    output_alphas: list[str],
) -> dict[str, str]:
    updated = dict(metadata) if metadata is not None else {}
    comment = updated.get("ss_training_comment", "")
    network_args = build_network_args(metadata, args)

    updated["ss_network_module"] = "networks.lora"
    updated["ss_network_args"] = json.dumps(network_args)

    if args.dynamic_method is None:
        updated["ss_network_dim"] = unify_metadata_scalars(output_ranks, str(args.rank))
        updated["ss_network_alpha"] = unify_metadata_scalars(output_alphas, format_metadata_scalar(None if args.alpha == "none" else args.rank))
        conversion_comment = f"converted from LoKr to LoRA, rank {args.rank}, alpha {args.alpha}"
    else:
        updated["ss_network_dim"] = "Dynamic"
        updated["ss_network_alpha"] = "Dynamic"
        conversion_comment = f"Dynamic convert from LoKr to LoRA with {args.dynamic_method}: {args.dynamic_param}, max rank {args.rank}"

    updated["ss_training_comment"] = f"{conversion_comment}; {comment}" if comment else conversion_comment
    return {k: str(v) for k, v in updated.items()}


def rebuild_lokr_components(prefix: str, reader: safe_open, key_set: set[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    def get_tensor(name: str) -> Optional[torch.Tensor]:
        key = f"{prefix}.{name}"
        if key not in key_set:
            return None
        return reader.get_tensor(key).to(device=device, dtype=torch.float32)

    alpha_tensor = get_tensor("alpha")
    lokr_w1 = get_tensor("lokr_w1")
    lokr_w2 = get_tensor("lokr_w2")
    lokr_w1_a = get_tensor("lokr_w1_a")
    lokr_w1_b = get_tensor("lokr_w1_b")
    lokr_w2_a = get_tensor("lokr_w2_a")
    lokr_w2_b = get_tensor("lokr_w2_b")
    lokr_t2 = get_tensor("lokr_t2")

    direct_alpha_ignored = False
    dim = None

    if lokr_w1 is None:
        if lokr_w1_a is None or lokr_w1_b is None:
            raise ValueError(f"Incomplete LoKr w1 tensors for '{prefix}'")
        dim = lokr_w1_b.shape[0]
        lokr_w1 = torch.mm(lokr_w1_a, lokr_w1_b)

    if lokr_w2 is None:
        if lokr_w2_a is None or lokr_w2_b is None:
            raise ValueError(f"Incomplete LoKr w2 tensors for '{prefix}'")
        dim = lokr_w2_b.shape[0]
        if lokr_t2 is None:
            lokr_w2 = torch.mm(lokr_w2_a, lokr_w2_b)
        else:
            lokr_w2 = torch.einsum(
                "i j k l, j r, i p -> p r k l",
                lokr_t2,
                lokr_w2_b,
                lokr_w2_a,
            )

    if alpha_tensor is not None and dim is not None:
        effective_scale = alpha_tensor.item() / dim
    else:
        effective_scale = 1.0
        direct_alpha_ignored = alpha_tensor is not None and dim is None

    if lokr_w1.ndim != 2:
        raise ValueError(f"Expected '{prefix}.lokr_w1' to rebuild into a 2D tensor, got {tuple(lokr_w1.shape)}")
    if lokr_w2.ndim < 2:
        raise ValueError(f"Expected '{prefix}.lokr_w2' to rebuild into at least 2 dims, got {tuple(lokr_w2.shape)}")

    return lokr_w1, lokr_w2, float(effective_scale), direct_alpha_ignored


def stable_rank(singular_values: torch.Tensor) -> int:
    if singular_values.numel() == 0:
        return 0
    max_sv = singular_values.max().item()
    if max_sv == 0.0:
        return 0
    tol = torch.finfo(singular_values.dtype).eps * max(singular_values.shape[0], 1) * max_sv
    return int((singular_values > tol).sum().item())


def index_sv_cumulative(singular_values: torch.Tensor, target: float) -> int:
    original_sum = float(torch.sum(singular_values))
    if original_sum <= 0.0:
        return 0
    cumulative_sums = torch.cumsum(singular_values, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, torch.tensor(target, device=singular_values.device, dtype=singular_values.dtype)).item())
    return max(0, min(index, len(singular_values) - 1))


def index_sv_fro(singular_values: torch.Tensor, target: float) -> int:
    singular_values_squared = singular_values.pow(2)
    fro_sq = float(torch.sum(singular_values_squared))
    if fro_sq <= 0.0:
        return 0
    cumulative_fro = torch.cumsum(singular_values_squared, dim=0) / fro_sq
    index = int(
        torch.searchsorted(
            cumulative_fro,
            torch.tensor(target**2, device=singular_values.device, dtype=singular_values.dtype),
        ).item()
    )
    return max(0, min(index, len(singular_values) - 1))


def index_sv_ratio(singular_values: torch.Tensor, target: float) -> int:
    max_sv = singular_values[0]
    min_sv = max_sv / target
    index = int(torch.sum(singular_values > min_sv).item()) - 1
    return max(0, min(index, len(singular_values) - 1))


def choose_rank_from_singular_values(
    singular_values: torch.Tensor,
    max_rank: int,
    dynamic_method: Optional[str],
    dynamic_param: Optional[float],
) -> int:
    if singular_values.numel() == 0:
        return 0

    if dynamic_method == "sv_ratio":
        new_rank = index_sv_ratio(singular_values, dynamic_param) + 1
    elif dynamic_method == "sv_cumulative":
        new_rank = index_sv_cumulative(singular_values, dynamic_param) + 1
    elif dynamic_method == "sv_fro":
        new_rank = index_sv_fro(singular_values, dynamic_param) + 1
    else:
        new_rank = max_rank

    if singular_values[0] <= MIN_SV:
        new_rank = 1
    elif new_rank > max_rank:
        new_rank = max_rank

    return max(1, min(new_rank, singular_values.numel()))


def calculate_retention_stats(singular_values: torch.Tensor, new_rank: int) -> tuple[float, float, float]:
    retained_sum = torch.sum(torch.abs(singular_values[:new_rank]))
    total_sum = torch.sum(torch.abs(singular_values))
    sum_retained = float(retained_sum / total_sum) if float(total_sum) > 0.0 else float("nan")

    singular_values_squared = singular_values.pow(2)
    fro_total = torch.sqrt(torch.sum(singular_values_squared))
    fro_retained = torch.sqrt(torch.sum(singular_values_squared[:new_rank]))
    fro_percent = float(fro_retained / fro_total) if float(fro_total) > 0.0 else float("nan")

    last_sv = singular_values[new_rank - 1]
    max_ratio = float(singular_values[0] / last_sv) if float(last_sv) > 0.0 else float("inf")
    return sum_retained, fro_percent, max_ratio


def prepare_kron_svd(
    lokr_w1: torch.Tensor,
    lokr_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
    out_l, in_m = lokr_w1.shape
    out_k, in_n = lokr_w2.shape[:2]
    kernel_shape = tuple(lokr_w2.shape[2:])
    lokr_w2_flat = lokr_w2.reshape(out_k, -1)

    u1, s1, vh1 = torch.linalg.svd(lokr_w1, full_matrices=False)
    u2, s2, vh2 = torch.linalg.svd(lokr_w2_flat, full_matrices=False)

    rank1 = stable_rank(s1)
    rank2 = stable_rank(s2)
    if rank1 == 0 or rank2 == 0:
        return u1[:, :0], s1[:0], vh1[:0, :], u2[:, :0], s2[:0], vh2[:0, :], kernel_shape, in_m, in_n

    u1 = u1[:, :rank1]
    s1 = s1[:rank1]
    vh1 = vh1[:rank1, :]
    u2 = u2[:, :rank2]
    s2 = s2[:rank2]
    vh2 = vh2[:rank2, :]

    return u1, s1, vh1, u2, s2, vh2, kernel_shape, in_m, in_n


def kron_svd_factors(
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int, int],
    requested_rank: int,
    matrix_scale: float,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    u1, s1, vh1, u2, s2, vh2, kernel_shape, in_m, in_n = prepared

    if s1.numel() == 0 or s2.numel() == 0 or matrix_scale == 0.0:
        return None, None, 0

    kron_singulars = torch.outer(s1, s2).reshape(-1)
    actual_rank = min(requested_rank, kron_singulars.numel())
    if actual_rank == 0:
        return None, None, 0

    values, indices = torch.topk(kron_singulars, k=actual_rank, largest=True, sorted=True)
    rank2 = s2.numel()
    j_idx = indices % rank2
    i_idx = torch.div(indices, rank2, rounding_mode="floor")

    up = torch.empty((u1.shape[0] * u2.shape[0], actual_rank), device=u1.device, dtype=torch.float32)
    down = torch.empty((actual_rank, vh1.shape[1] * vh2.shape[1]), device=u1.device, dtype=torch.float32)

    scale_sign = -1.0 if matrix_scale < 0 else 1.0
    scale_abs = abs(matrix_scale)

    # kron(A, B) has an SVD assembled directly from the singular vectors/values of A and B.
    for column, (i, j, sigma) in enumerate(zip(i_idx.tolist(), j_idx.tolist(), values.tolist())):
        up[:, column] = torch.kron(u1[:, i], u2[:, j]) * (sigma * scale_abs * scale_sign)
        down[column, :] = torch.kron(vh1[i, :], vh2[j, :])

    if kernel_shape:
        up = up.reshape(up.shape[0], up.shape[1], *([1] * len(kernel_shape)))
        down = down.reshape(actual_rank, in_m * in_n, *kernel_shape)

    return up, down, actual_rank


def convert_alpha_mode(alpha_arg: str, actual_rank: int) -> tuple[Optional[float], float]:
    if actual_rank <= 0:
        return None, 1.0

    if alpha_arg == "none":
        return None, 1.0
    if alpha_arg == "rank":
        return float(actual_rank), 1.0

    alpha_value = float(alpha_arg)
    if alpha_value == 0.0:
        raise ValueError("numeric alpha must be non-zero")
    return alpha_value, alpha_value / actual_rank


def main() -> None:
    args = parse_args()
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if args.dynamic_method is not None and args.dynamic_param is None:
        raise ValueError("If using --dynamic_method, then --dynamic_param is required")
    if args.dynamic_method is None and args.dynamic_param is not None:
        raise ValueError("--dynamic_param requires --dynamic_method")
    if args.input.suffix.lower() != ".safetensors":
        raise ValueError("input must be a .safetensors file")
    if args.output.suffix.lower() != ".safetensors":
        raise ValueError("output must be a .safetensors file")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")

    device = resolve_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with safe_open(str(args.input), framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        key_set = set(keys)
        lokr_prefixes = build_lokr_prefixes(keys)
        if not lokr_prefixes:
            raise ValueError("No LoKr tensors were found in the input file")

        source_metadata = reader.metadata()
        output_sd: dict[str, torch.Tensor] = {}

        converted = 0
        skipped_zero = 0
        copied = 0
        direct_alpha_ignored = 0
        output_ranks: list[str] = []
        output_alphas: list[str] = []
        fro_list: list[float] = []

        if args.dynamic_method:
            print(
                f"Dynamically determining new alphas and dims based off {args.dynamic_method}: "
                f"{args.dynamic_param}, max rank is {args.rank}"
            )

        for prefix in tqdm(sorted(lokr_prefixes)):
            lokr_w1, lokr_w2, input_scale, alpha_ignored = rebuild_lokr_components(prefix, reader, key_set, device)
            prepared = prepare_kron_svd(lokr_w1, lokr_w2)
            singular_values = torch.sort(torch.outer(prepared[1], prepared[4]).reshape(-1), descending=True).values
            actual_rank = choose_rank_from_singular_values(singular_values, args.rank, args.dynamic_method, args.dynamic_param)

            if actual_rank == 0:
                skipped_zero += 1
                if alpha_ignored:
                    direct_alpha_ignored += 1
                continue

            alpha_value, output_scale = convert_alpha_mode(args.alpha, actual_rank)
            factor_scale = input_scale / output_scale
            up, down, actual_rank = kron_svd_factors(prepared, actual_rank, factor_scale)
            if actual_rank == 0 or up is None or down is None:
                skipped_zero += 1
                if alpha_ignored:
                    direct_alpha_ignored += 1
                continue

            out_dtype = output_dtype_for(prefix, reader, key_set, args.save_precision)
            output_sd[f"{prefix}.lora_up.weight"] = up.to(dtype=out_dtype, device="cpu").contiguous()
            output_sd[f"{prefix}.lora_down.weight"] = down.to(dtype=out_dtype, device="cpu").contiguous()

            if alpha_value is not None:
                output_sd[f"{prefix}.alpha"] = torch.tensor(alpha_value, dtype=out_dtype)
                output_alphas.append(format_metadata_scalar(alpha_value))
            else:
                output_alphas.append("None")

            output_ranks.append(str(actual_rank))

            sum_retained, fro_retained, max_ratio = calculate_retention_stats(singular_values, actual_rank)
            if not np.isnan(fro_retained):
                fro_list.append(float(fro_retained))

            if args.verbose:
                verbose_str = f"{prefix:75} | "
                verbose_str += f"sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"
                if args.dynamic_method:
                    verbose_str += f", dynamic | dim: {actual_rank}, alpha: {output_alphas[-1]}"
                tqdm.write(verbose_str)

            dora_key = f"{prefix}.dora_scale"
            if dora_key in key_set:
                output_sd[dora_key] = reader.get_tensor(dora_key)

            converted += 1
            if alpha_ignored:
                direct_alpha_ignored += 1

        for key in keys:
            prefix, suffix = split_key_suffix(key)
            if prefix in lokr_prefixes and suffix in (LOKR_SUFFIXES | SHARED_SUFFIXES):
                continue
            output_sd[key] = reader.get_tensor(key)
            copied += 1

    metadata = metadata_for_output(source_metadata, args, output_ranks, output_alphas)
    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(output_sd, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    save_file(output_sd, str(args.output), metadata=metadata)

    if args.verbose and fro_list:
        print(f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}")
    print(f"Converted {converted} LoKr modules into LoRA modules at rank <= {args.rank}.")
    print(f"Skipped {skipped_zero} zero-effect modules.")
    print(f"Copied {copied} non-LoKr tensors unchanged.")
    print(f"Used device: {device}.")
    if direct_alpha_ignored:
        print(
            "Note: "
            f"{direct_alpha_ignored} direct LoKr modules had .alpha tensors, but Comfy's current "
            "LoKr loader ignores alpha when the file stores direct lokr_w1/lokr_w2 weights. "
            "The conversion mirrors that effective behavior."
        )


if __name__ == "__main__":
    main()
