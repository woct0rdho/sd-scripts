#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from safetensors.torch import load_file

import sdxl_train_control_net as control_net_train
import library.args as args_util
from library import sdxl_model_util
from library.utils import setup_logging

setup_logging()

import logging

logger = logging.getLogger(__name__)

OriginalSdxlControlNet = control_net_train.SdxlControlNet
OriginalSdxlControlledUNet = control_net_train.SdxlControlledUNet
OriginalTrainControlNetTrain = control_net_train.train

CONTROL_LORA_TARGET_PREFIXES = ("time_embed.", "label_emb.", "input_blocks.", "middle_block.")


@dataclass
class ControlLoraTrainContext:
    rank: int
    ranks: dict[str, int]
    dropout: Optional[float]
    train_bias_norm: bool
    share_base_weights: bool
    network_weights: Optional[str]
    extract_device: str
    need_base_unet_sd: bool
    base_unet_sd: Optional[dict[str, torch.Tensor]] = None
    base_unet: Optional[torch.nn.Module] = None


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_weights_file(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return load_file(path, device="cpu")

    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        return loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"expected a state dict in {path}, got {type(loaded)}")
    return loaded


def is_control_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return "lora_controlnet" in state_dict


def extract_lora(diff: torch.Tensor, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    conv2d = diff.ndim == 4
    kernel_size = None if not conv2d else diff.shape[2:4]
    out_dim, in_dim = diff.shape[0:2]
    rank = min(rank, in_dim, out_dim)
    if rank < 1:
        raise ValueError(f"invalid rank {rank} for tensor shape {tuple(diff.shape)}")

    if conv2d:
        if kernel_size == (1, 1):
            mat = diff.reshape(out_dim, in_dim)
        else:
            mat = diff.flatten(start_dim=1)
    else:
        mat = diff

    mat = mat.to(device=device, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    S = S[:rank]

    sign_S = torch.sign(S)
    sqrt_abs_S = torch.sqrt(torch.abs(S))
    up = U[:, :rank] * (sign_S * sqrt_abs_S)
    down = sqrt_abs_S[:, None] * Vh[:rank, :]

    if conv2d:
        up = up.reshape(out_dim, rank, 1, 1)
        down = down.reshape(rank, in_dim, kernel_size[0], kernel_size[1])

    return up.cpu(), down.cpu()


class ControlLoRAModule(torch.nn.Module):
    def __init__(self, lora_name: str, org_module: torch.nn.Module, rank: int, dropout: Optional[float]) -> None:
        super().__init__()
        self.lora_name = lora_name
        self.org_module = org_module
        self.dropout = dropout

        if isinstance(org_module, torch.nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            rank = min(rank, in_dim, out_dim)
            self.lora_down = torch.nn.Conv2d(
                in_dim,
                rank,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                dilation=org_module.dilation,
                bias=False,
            )
            self.lora_up = torch.nn.Conv2d(rank, out_dim, (1, 1), (1, 1), bias=False)
        elif isinstance(org_module, torch.nn.Linear):
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            rank = min(rank, in_dim, out_dim)
            self.lora_down = torch.nn.Linear(in_dim, rank, bias=False)
            self.lora_up = torch.nn.Linear(rank, out_dim, bias=False)
        else:
            raise TypeError(f"unsupported module for ControlLoRA: {org_module.__class__.__name__}")

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_up.weight)

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lx = self.lora_down(x)
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        return self.org_forward(x) + self.lora_up(lx)


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


def convert_comfy_controlnet_to_sdxl_load_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    unet_sd: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith(("input_hint_block.", "zero_convs.", "middle_block_out.")):
            converted[comfy_key_to_controlnet(key)] = value
        elif key.startswith(CONTROL_LORA_TARGET_PREFIXES) or key.startswith("output_blocks.") or key.startswith("out."):
            unet_sd[key] = value
        else:
            converted[key] = value

    if unet_sd:
        converted.update(sdxl_model_util.convert_sdxl_unet_state_dict_to_diffusers(unet_sd))
    return converted


def share_control_lora_base_weights(
    control_root: torch.nn.Module,
    base_root: torch.nn.Module,
    lora_names: set[str],
) -> int:
    shared = 0
    for name in sorted(lora_names):
        control_module = control_root.get_submodule(name)
        base_module = base_root.get_submodule(name)

        control_weight = getattr(control_module, "weight", None)
        base_weight = getattr(base_module, "weight", None)
        if control_weight is None or base_weight is None:
            continue
        if control_weight.shape != base_weight.shape:
            raise ValueError(
                f"cannot share ControlLoRA base weight {name}: control {tuple(control_weight.shape)} "
                f"!= base {tuple(base_weight.shape)}"
            )

        base_weight.requires_grad_(False)
        control_module.weight = base_weight
        shared += 1
    return shared


def make_control_lora_model_classes(context: ControlLoraTrainContext):
    class CapturingSdxlControlledUNet(OriginalSdxlControlledUNet):
        def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = True):
            if context.need_base_unet_sd:
                context.base_unet_sd = {
                    k: v.detach().to("cpu")
                    for k, v in state_dict.items()
                    if isinstance(v, torch.Tensor) and not k.startswith("out")
                }
                logger.info(f"captured base UNet tensors for ControlLoRA extraction: {len(context.base_unet_sd)}")
            info = super().load_state_dict(state_dict, strict=strict, assign=assign)
            context.base_unet = self
            return info

        def forward(self, x, timesteps=None, context=None, y=None, input_resi_add=None, mid_add=None, **kwargs):
            if mid_add is not None and mid_add.dtype != x.dtype:
                mid_add = mid_add.to(dtype=x.dtype)
            if input_resi_add is not None:
                input_resi_add = [resi.to(dtype=x.dtype) if resi.dtype != x.dtype else resi for resi in input_resi_add]
            return super().forward(x, timesteps, context, y, input_resi_add, mid_add, **kwargs)

    class SdxlControlLoRANet(OriginalSdxlControlNet):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.control_lora_modules = torch.nn.ModuleList()
            self._control_lora_applied = False
            self._control_lora_names: set[str] = set()

        def init_from_unet(self, unet):
            info = super().init_from_unet(unet)
            self.apply_control_lora(unet.state_dict(), initialize_from_current=False, base_unet=unet)
            if context.network_weights is not None:
                self.load_control_lora_weights(context.network_weights)
            return info

        def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = True):
            if is_control_lora_state_dict(state_dict):
                raise ValueError(
                    "--controlnet_model_name_or_path was given a ControlLoRA file. "
                    "Use --network_weights for ControlLoRA weights, or pass a full ControlNet here."
                )

            if any(k.startswith("control_model.") for k in state_dict):
                state_dict = {k[len("control_model.") :] if k.startswith("control_model.") else k: v for k, v in state_dict.items()}

            if any(k.startswith(("input_hint_block.", "zero_convs.", "middle_block_out.")) for k in state_dict):
                state_dict = convert_comfy_controlnet_to_sdxl_load_format(dict(state_dict))

            info = OriginalSdxlControlNet.load_state_dict(self, state_dict, strict=strict, assign=assign)
            if context.base_unet_sd is None:
                raise ValueError("internal error: base UNet state dict was not captured before loading full ControlNet")

            self.apply_control_lora(context.base_unet_sd, initialize_from_current=True, base_unet=context.base_unet)
            return info

        def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
            for name, param in torch.nn.Module.named_parameters(self, prefix, recurse, remove_duplicate):
                if param.requires_grad:
                    yield name, param

        def apply_control_lora(
            self,
            base_unet_sd: dict[str, torch.Tensor],
            initialize_from_current: bool,
            base_unet: Optional[torch.nn.Module],
        ) -> None:
            if self._control_lora_applied:
                return

            logger.info(f"applying ControlLoRA modules: rank={context.rank}, dropout={context.dropout}")
            modules: list[ControlLoRAModule] = []
            device = resolve_device(context.extract_device)

            for name, module in list(torch.nn.Module.named_modules(self)):
                if not name.startswith(CONTROL_LORA_TARGET_PREFIXES):
                    continue
                if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    continue

                weight_key = f"{name}.weight"
                if weight_key not in base_unet_sd:
                    continue

                rank = context.ranks.get(name, context.rank)
                lora = ControlLoRAModule(name, module, rank, context.dropout)
                lora.to(device=module.weight.device, dtype=module.weight.dtype)

                if initialize_from_current:
                    base_weight = base_unet_sd[weight_key].to(device=module.weight.device, dtype=module.weight.dtype)
                    diff = module.weight.detach().float().cpu() - base_unet_sd[weight_key].float().cpu()
                    up, down = extract_lora(diff, rank, device)
                    module.weight.data.copy_(base_weight)
                    lora.lora_up.weight.data.copy_(up.to(device=module.weight.device, dtype=module.weight.dtype))
                    lora.lora_down.weight.data.copy_(down.to(device=module.weight.device, dtype=module.weight.dtype))

                lora.apply_to()
                modules.append(lora)
                self._control_lora_names.add(name)

            self.control_lora_modules = torch.nn.ModuleList(modules)
            self._control_lora_applied = True

            if context.share_base_weights:
                if base_unet is None:
                    raise ValueError("cannot share ControlLoRA base weights because the base UNet module is unavailable")
                shared = share_control_lora_base_weights(self, base_unet, self._control_lora_names)
                if context.train_bias_norm:
                    logger.info("shared ControlLoRA base weight tensors; bias/norm tensors remain independent and trainable")
                logger.info(f"shared ControlLoRA base weights: {shared}")

            self.prepare_control_lora_params()
            logger.info(f"enabled ControlLoRA for {len(self.control_lora_modules)} modules")

        def prepare_control_lora_params(self) -> None:
            torch.nn.Module.requires_grad_(self, False)

            for module in self.control_lora_modules:
                module.requires_grad_(True)

            for name, param in torch.nn.Module.named_parameters(self):
                if name.startswith("controlnet_"):
                    param.requires_grad_(True)
                elif context.train_bias_norm and name.startswith(CONTROL_LORA_TARGET_PREFIXES) and param.ndim < 2:
                    param.requires_grad_(True)

            trainable = sum(p.numel() for _, p in self.named_parameters())
            logger.info(f"ControlLoRA trainable parameters: {trainable}")

        def load_control_lora_weights(self, file: str) -> None:
            logger.info(f"loading ControlLoRA weights: {file}")
            weights_sd = load_weights_file(file)
            if not is_control_lora_state_dict(weights_sd):
                raise ValueError(f"{file} is not a ControlLoRA file: missing lora_controlnet marker")

            lora_by_name = {module.lora_name: module for module in self.control_lora_modules}
            loaded_loras = 0
            for name, module in lora_by_name.items():
                up_key = f"{name}.up"
                down_key = f"{name}.down"
                if up_key in weights_sd and down_key in weights_sd:
                    if module.lora_up.weight.shape != weights_sd[up_key].shape:
                        raise ValueError(
                            f"ControlLoRA rank/shape mismatch for {up_key}: checkpoint {tuple(weights_sd[up_key].shape)} "
                            f"!= model {tuple(module.lora_up.weight.shape)}"
                        )
                    if module.lora_down.weight.shape != weights_sd[down_key].shape:
                        raise ValueError(
                            f"ControlLoRA rank/shape mismatch for {down_key}: checkpoint {tuple(weights_sd[down_key].shape)} "
                            f"!= model {tuple(module.lora_down.weight.shape)}"
                        )
                    module.lora_up.weight.data.copy_(
                        weights_sd[up_key].to(device=module.lora_up.weight.device, dtype=module.lora_up.weight.dtype)
                    )
                    module.lora_down.weight.data.copy_(
                        weights_sd[down_key].to(device=module.lora_down.weight.device, dtype=module.lora_down.weight.dtype)
                    )
                    loaded_loras += 1

            raw_sd = torch.nn.Module.state_dict(self)
            loaded_full = 0
            for key, value in weights_sd.items():
                if key == "lora_controlnet" or key.endswith(".up") or key.endswith(".down") or key.endswith(".alpha"):
                    continue
                raw_key = comfy_key_to_controlnet(key)
                if raw_key not in raw_sd:
                    logger.warning(f"ControlLoRA key not found in model, skipping: {key}")
                    continue
                if raw_sd[raw_key].shape != value.shape:
                    logger.warning(
                        f"ControlLoRA shape mismatch for {key}: checkpoint {tuple(value.shape)} != model {tuple(raw_sd[raw_key].shape)}"
                    )
                    continue
                raw_sd[raw_key] = value.to(dtype=raw_sd[raw_key].dtype)
                loaded_full += 1

            torch.nn.Module.load_state_dict(self, raw_sd, strict=False)
            self.prepare_control_lora_params()
            logger.info(f"loaded ControlLoRA modules: {loaded_loras}, full tensors: {loaded_full}")

        def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
            if not self._control_lora_applied:
                return OriginalSdxlControlNet.state_dict(self, destination=destination, prefix=prefix, keep_vars=keep_vars)

            raw_sd = torch.nn.Module.state_dict(self, destination=None, prefix="", keep_vars=keep_vars)
            output_sd: dict[str, torch.Tensor] = {}

            for module in self.control_lora_modules:
                output_sd[f"{module.lora_name}.up"] = module.lora_up.weight.detach().clone()
                output_sd[f"{module.lora_name}.down"] = module.lora_down.weight.detach().clone()

            for key, value in raw_sd.items():
                if key.startswith("control_lora_modules."):
                    continue

                if key.startswith("controlnet_"):
                    output_sd[controlnet_key_to_comfy(key)] = value.detach().clone()
                    continue

                if context.train_bias_norm and key.startswith(CONTROL_LORA_TARGET_PREFIXES) and value.ndim < 2:
                    output_sd[key] = value.detach().clone()

            output_sd["lora_controlnet"] = torch.tensor([])

            if prefix:
                output_sd = {prefix + k: v for k, v in output_sd.items()}
            if destination is not None:
                destination.update(output_sd)
                return destination
            return output_sd

    return SdxlControlLoRANet, CapturingSdxlControlledUNet


def setup_parser() -> argparse.ArgumentParser:
    parser = control_net_train.setup_parser()
    parser.description = "Train a ComfyUI-format SDXL ControlLoRA."

    parser.add_argument("--network_dim", type=int, default=256, help="ControlLoRA rank / ControlLoRAのrank")
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Dropout for ControlLoRA hidden activations / ControlLoRAのdropout",
    )
    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="Existing ControlLoRA weights to continue training / 追加学習するControlLoRA重み",
    )
    parser.add_argument(
        "--lora_lr",
        type=float,
        default=None,
        help="Learning rate for ControlLoRA and bias/norm tensors. Defaults to --learning_rate.",
    )
    parser.add_argument(
        "--control_lora_extract_device",
        type=str,
        default="auto",
        help="Device for SVD when initializing from a full ControlNet: auto, cpu, cuda, ...",
    )
    parser.add_argument(
        "--control_lora_no_train_bias_norm",
        action="store_true",
        help="Do not train/save copied UNet bias and norm tensors; train only LoRA and ControlNet-only modules.",
    )
    parser.add_argument(
        "--share_control_lora_base_weights",
        action="store_true",
        help="Share frozen ControlNet trunk weight tensors with the generation UNet to reduce parameter VRAM.",
    )
    return parser


def build_context(args: argparse.Namespace) -> ControlLoraTrainContext:
    if args.network_dim is None or args.network_dim < 1:
        raise ValueError("--network_dim must be at least 1")
    if args.network_weights is not None and args.controlnet_model_name_or_path is not None:
        raise ValueError("Use either --network_weights (ControlLoRA) or --controlnet_model_name_or_path (full ControlNet), not both.")
    if args.save_state or args.save_state_on_train_end:
        raise ValueError("--save_state is not supported yet by sdxl_train_control_lora.py")
    if args.resume is not None:
        raise ValueError("--resume is not supported yet by sdxl_train_control_lora.py")
    if args.lora_lr is not None:
        args.learning_rate = args.lora_lr

    ranks: dict[str, int] = {}
    if args.network_weights is not None:
        weights_sd = load_weights_file(args.network_weights)
        if not is_control_lora_state_dict(weights_sd):
            raise ValueError(f"--network_weights must be a ControlLoRA file with lora_controlnet marker: {args.network_weights}")
        ranks = {key[: -len(".down")]: int(value.shape[0]) for key, value in weights_sd.items() if key.endswith(".down")}
        logger.info(f"using per-layer ranks from --network_weights: {len(ranks)} modules")

    return ControlLoraTrainContext(
        rank=args.network_dim,
        ranks=ranks,
        dropout=args.network_dropout,
        train_bias_norm=not args.control_lora_no_train_bias_norm,
        share_base_weights=args.share_control_lora_base_weights,
        network_weights=args.network_weights,
        extract_device=args.control_lora_extract_device,
        need_base_unet_sd=args.controlnet_model_name_or_path is not None,
    )


def train(args: argparse.Namespace) -> None:
    context = build_context(args)
    control_lora_cls, controlled_unet_cls = make_control_lora_model_classes(context)

    original_controlnet_cls = control_net_train.SdxlControlNet
    original_controlled_unet_cls = control_net_train.SdxlControlledUNet
    try:
        control_net_train.SdxlControlNet = control_lora_cls
        control_net_train.SdxlControlledUNet = controlled_unet_cls
        OriginalTrainControlNetTrain(args)
    finally:
        control_net_train.SdxlControlNet = original_controlnet_cls
        control_net_train.SdxlControlledUNet = original_controlled_unet_cls


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)
    train(args)
