#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from library import sdxl_model_util
from library.utils import add_logging_arguments, setup_logging

setup_logging()

import logging

logger = logging.getLogger(__name__)

SAVE_PRECISION_MAP = {
    "float": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

SDXL_UNET_PREFIXES = (
    "time_embed.",
    "label_emb.",
    "input_blocks.",
    "middle_block.",
    "output_blocks.",
    "out.",
)

DIFFUSERS_UNET_PREFIXES = (
    "conv_in.",
    "time_embedding.",
    "add_embedding.",
    "down_blocks.",
    "mid_block.",
    "up_blocks.",
    "conv_norm_out.",
    "conv_out.",
)

CONTROLNET_SD_PREFIXES = (
    "input_hint_block.",
    "zero_convs.",
    "middle_block_out.",
)

CONTROLNET_DIFFUSERS_PREFIXES = (
    "controlnet_cond_embedding.",
    "controlnet_down_blocks.",
    "controlnet_mid_block.",
)


def str_to_dtype(value: Optional[str]) -> Optional[torch.dtype]:
    if value is None or value == "preserve":
        return None
    return SAVE_PRECISION_MAP[value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a ComfyUI ControlLoRA from a full SDXL ControlNet and the "
            "corresponding base SDXL checkpoint. The output uses keys like "
            "input_blocks.1.0.in_layers.2.up/.down plus a lora_controlnet marker."
        )
    )
    parser.add_argument(
        "--base_model",
        required=True,
        type=Path,
        help="Base SDXL checkpoint or UNet state dict used to initialize the ControlNet.",
    )
    parser.add_argument(
        "--controlnet_model",
        required=True,
        type=Path,
        help="Full SDXL ControlNet checkpoint to approximate as ControlLoRA.",
    )
    parser.add_argument("--save_to", required=True, type=Path, help="Output .safetensors or .pt file.")
    parser.add_argument("--rank", type=int, default=256, help="Maximum ControlLoRA rank for each extracted weight.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for SVD work: auto, cpu, cuda, cuda:0, ... Default: auto.",
    )
    parser.add_argument(
        "--save_precision",
        choices=("float", "fp16", "bf16"),
        default="bf16",
        help="Precision for extracted up/down tensors and fp32 copied tensors. Default: bf16.",
    )
    parser.add_argument(
        "--full_weight_precision",
        choices=("preserve", "float", "fp16", "bf16"),
        default="preserve",
        help=(
            "Precision for copied non-LoRA tensors such as input_hint_block and biases. "
            "'preserve' keeps source dtype except fp32 tensors are cast to --save_precision. Default: preserve."
        ),
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Write minimal safetensors metadata. By default metadata is omitted to match ComfyUI's ControlLoraSave node.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-layer Frobenius retention.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting --save_to.")
    add_logging_arguments(parser)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)

    logger.info(f"loading: {path}")
    if path.suffix.lower() == ".safetensors":
        sd = load_file(str(path), device="cpu")
    else:
        loaded = torch.load(str(path), map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            sd = loaded["state_dict"]
        else:
            sd = loaded

    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(sd)}")

    return {str(k): v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def strip_prefix(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def has_any_prefix(keys: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(any(k.startswith(prefix) for prefix in prefixes) for k in keys)


def comfy_key_to_controlnet(key: str) -> str:
    if key.startswith("input_hint_block."):
        parts = key.split(".")
        index = int(parts[1])
        suffix = ".".join(parts[2:])
        if index == 0:
            return f"controlnet_cond_embedding.conv_in.{suffix}"
        if index == 14:
            return f"controlnet_cond_embedding.conv_out.{suffix}"
        if index % 2 == 0:
            return f"controlnet_cond_embedding.blocks.{index // 2 - 1}.{suffix}"
    if key.startswith("zero_convs."):
        parts = key.split(".")
        if len(parts) >= 4 and parts[2] == "0":
            return "controlnet_down_blocks.{}.{}".format(parts[1], ".".join(parts[3:]))
    if key.startswith("middle_block_out.0."):
        return key.replace("middle_block_out.0.", "controlnet_mid_block.", 1)
    return key


def controlnet_key_to_comfy(key: str) -> str:
    if key.startswith("controlnet_cond_embedding.conv_in."):
        return key.replace("controlnet_cond_embedding.conv_in.", "input_hint_block.0.", 1)
    if key.startswith("controlnet_cond_embedding.blocks."):
        parts = key.split(".")
        block_index = int(parts[2])
        return "input_hint_block.{}.{}".format((block_index + 1) * 2, ".".join(parts[3:]))
    if key.startswith("controlnet_cond_embedding.conv_out."):
        return key.replace("controlnet_cond_embedding.conv_out.", "input_hint_block.14.", 1)
    if key.startswith("controlnet_down_blocks."):
        parts = key.split(".")
        block_index = int(parts[1])
        return "zero_convs.{}.0.{}".format(block_index, ".".join(parts[2:]))
    if key.startswith("controlnet_mid_block."):
        return key.replace("controlnet_mid_block.", "middle_block_out.0.", 1)
    return key


def convert_diffusers_controlnet_to_comfy(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    unet_sd: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith(CONTROLNET_DIFFUSERS_PREFIXES):
            converted[controlnet_key_to_comfy(key)] = value
        elif key.startswith(DIFFUSERS_UNET_PREFIXES):
            unet_sd[key] = value
        else:
            converted[key] = value

    if unet_sd:
        converted.update(sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl(unet_sd))
    return converted


def extract_base_unet_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        unet_sd = strip_prefix(sd, prefix)
        if unet_sd:
            return unet_sd

    unet_prefixed = strip_prefix(sd, "unet.")
    if unet_prefixed:
        return extract_base_unet_state_dict(unet_prefixed)

    keys = set(sd.keys())
    if has_any_prefix(keys, SDXL_UNET_PREFIXES):
        return {k: v for k, v in sd.items() if k.startswith(SDXL_UNET_PREFIXES)}

    if has_any_prefix(keys, DIFFUSERS_UNET_PREFIXES):
        diffusers_sd = {k: v for k, v in sd.items() if k.startswith(DIFFUSERS_UNET_PREFIXES)}
        return sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl(diffusers_sd)

    raise ValueError(
        "Could not find an SDXL UNet in --base_model. Expected model.diffusion_model.*, "
        "input_blocks.*, or diffusers UNet keys such as conv_in.weight."
    )


def normalize_controlnet_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "lora_controlnet" in sd:
        raise ValueError("--controlnet_model is already a ControlLoRA file, not a full ControlNet checkpoint.")

    if any(k.startswith("control_model.") for k in sd):
        sd = {k[len("control_model.") :] if k.startswith("control_model.") else k: v for k, v in sd.items()}

    keys = set(sd.keys())
    if has_any_prefix(keys, CONTROLNET_SD_PREFIXES) or has_any_prefix(keys, SDXL_UNET_PREFIXES):
        return sd

    if has_any_prefix(keys, CONTROLNET_DIFFUSERS_PREFIXES) or has_any_prefix(keys, DIFFUSERS_UNET_PREFIXES):
        return convert_diffusers_controlnet_to_comfy(dict(sd))

    raise ValueError(
        "Could not detect --controlnet_model format. Expected SD/Comfy ControlNet keys "
        "such as input_hint_block.*, or diffusers keys such as controlnet_cond_embedding.*."
    )


def cast_for_save(tensor: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if dtype is not None and tensor.dtype.is_floating_point:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().cpu().contiguous()


def cast_full_weight(tensor: torch.Tensor, save_dtype: torch.dtype, full_weight_dtype: Optional[torch.dtype]) -> torch.Tensor:
    if full_weight_dtype is not None:
        return cast_for_save(tensor, full_weight_dtype)

    # Match ComfyUI's ControlLoraSave behavior: preserve fp16/bf16 full weights,
    # but avoid writing large fp32 control modules unless explicitly requested.
    if tensor.dtype == torch.float32:
        return cast_for_save(tensor, save_dtype)
    return cast_for_save(tensor, None)


def extract_lora(diff: torch.Tensor, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    conv2d = diff.ndim == 4
    kernel_size = None if not conv2d else diff.shape[2:4]
    out_dim, in_dim = diff.shape[0:2]
    rank = min(rank, in_dim, out_dim)
    if rank < 1:
        raise ValueError(f"Invalid rank {rank} for tensor with shape {tuple(diff.shape)}")

    if conv2d:
        if kernel_size == (1, 1):
            mat = diff.reshape(out_dim, in_dim)
        else:
            mat = diff.flatten(start_dim=1)
    else:
        mat = diff

    mat = mat.to(device=device, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

    denom = torch.sum(S * S)
    S_rank = S[:rank]
    if denom.item() == 0:
        fro_ratio = 1.0
    else:
        fro_ratio = (torch.sum(S_rank * S_rank) / denom).item()

    sign_S = torch.sign(S_rank)
    sqrt_abs_S = torch.sqrt(torch.abs(S_rank))
    up = U[:, :rank] * (sign_S * sqrt_abs_S)
    down = sqrt_abs_S[:, None] * Vh[:rank, :]

    if conv2d:
        up = up.reshape(out_dim, rank, 1, 1)
        down = down.reshape(rank, in_dim, kernel_size[0], kernel_size[1])

    return up.cpu(), down.cpu(), fro_ratio


def output_lora_key(control_key: str, suffix: str) -> str:
    name = control_key
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return f"{name}.{suffix}"


def build_metadata(args: argparse.Namespace, extracted_count: int) -> Optional[dict[str, str]]:
    if not args.metadata:
        return None
    return {
        "ss_network_dim": str(args.rank),
        "ss_network_alpha": str(args.rank),
        "ss_training_comment": (
            "ControlLoRA extracted from full SDXL ControlNet; "
            f"rank {args.rank}; extracted tensors {extracted_count}"
        ),
    }


def save_state_dict(path: Path, state_dict: dict[str, torch.Tensor], metadata: Optional[dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace it.")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"saving: {path}")
    if path.suffix.lower() == ".safetensors":
        save_file(state_dict, str(path), metadata)
    else:
        torch.save(state_dict, str(path))


def extract_control_lora(args: argparse.Namespace) -> None:
    if args.rank < 1:
        raise ValueError("--rank must be at least 1")

    device = resolve_device(args.device)
    save_dtype = str_to_dtype(args.save_precision)
    assert save_dtype is not None
    full_weight_dtype = str_to_dtype(args.full_weight_precision)

    base_raw_sd = load_state_dict(args.base_model)
    base_unet_sd = extract_base_unet_state_dict(base_raw_sd)
    del base_raw_sd
    logger.info(f"base UNet tensors: {len(base_unet_sd)}")

    control_raw_sd = load_state_dict(args.controlnet_model)
    control_sd = normalize_controlnet_state_dict(control_raw_sd)
    del control_raw_sd
    logger.info(f"normalized ControlNet tensors: {len(control_sd)}")

    output_sd: dict[str, torch.Tensor] = {}
    stored: set[str] = set()
    extracted = 0
    copied_from_base_path = 0
    skipped_missing = 0
    skipped_shape = 0
    fro_sum = 0.0

    for key, base_weight in tqdm(list(base_unet_sd.items()), desc="extracting ControlLoRA"):
        if key not in control_sd:
            skipped_missing += 1
            continue

        control_weight = control_sd[key]
        if control_weight.shape != base_weight.shape:
            logger.warning(f"shape mismatch for {key}: control {tuple(control_weight.shape)} != base {tuple(base_weight.shape)}; copying full tensor")
            output_sd[key] = cast_full_weight(control_weight, save_dtype, full_weight_dtype)
            stored.add(key)
            skipped_shape += 1
            continue

        if control_weight.ndim >= 2 and control_weight.dtype.is_floating_point and base_weight.dtype.is_floating_point:
            diff = control_weight.float() - base_weight.float()
            up, down, fro_ratio = extract_lora(diff, args.rank, device)
            output_sd[output_lora_key(key, "up")] = cast_for_save(up, save_dtype)
            output_sd[output_lora_key(key, "down")] = cast_for_save(down, save_dtype)
            if args.verbose:
                tqdm.write(f"{key:80} fro_ratio {fro_ratio:.5f}")
            fro_sum += fro_ratio
            extracted += 1
        else:
            output_sd[key] = cast_full_weight(control_weight, save_dtype, full_weight_dtype)
            copied_from_base_path += 1

        stored.add(key)

    copied_control_only = 0
    for key, value in tqdm(list(control_sd.items()), desc="copying ControlNet-only tensors"):
        if key in stored:
            continue
        output_sd[key] = cast_full_weight(value, save_dtype, full_weight_dtype)
        copied_control_only += 1

    output_sd["lora_controlnet"] = torch.tensor([])

    logger.info(f"extracted LoRA weights: {extracted}")
    logger.info(f"copied matching non-LoRA tensors: {copied_from_base_path}")
    logger.info(f"copied ControlNet-only tensors: {copied_control_only}")
    logger.info(f"base tensors missing in ControlNet: {skipped_missing}")
    if skipped_shape:
        logger.info(f"shape mismatches copied as full tensors: {skipped_shape}")
    if extracted:
        logger.info(f"average Frobenius retention: {fro_sum / extracted:.5f}")
    logger.info(f"output tensors: {len(output_sd)}")

    metadata = build_metadata(args, extracted)
    save_state_dict(args.save_to, output_sd, metadata, args.overwrite)


def main() -> None:
    args = parse_args()
    setup_logging(args, reset=True)
    extract_control_lora(args)


if __name__ == "__main__":
    main()
