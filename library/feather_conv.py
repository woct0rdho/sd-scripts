"""Feather ops 3x3 stride-1 conv2d for the VAE, as a drop-in that keeps torch's fallbacks.

`feather_ops.conv2d` runs a HIP kernel for bf16 NCHW 3x3 stride-1 padding-1 convolutions and delegates
two cases to MIOpen, where the kernel loses: fp16 (MIOpen's Winograd does 4/9 of the multiplies) and the
shapes the kernel cannot tile (a VAE's first and last conv). Everything else falls back to `F.conv2d`.

VAEs are used in two places: encoding the training images (latent caching, `--cache_latents`) and
decoding the sample images (`--sample_every_n_steps`). Both go through the same model object, so the
wiring is a one-time relabel of its eligible `nn.Conv2d` modules (`patch_conv_modules`), which keeps the
parameters, buffers, module names and `state_dict` keys untouched - the forward of those modules only
gains a fast path in front of `super().forward()`.

The fast path only runs for a frozen bf16/fp16 VAE on a HIP device under `no_grad`, and only for a
contiguous NCHW input. `ChunkedConv2d`-style modules (sd-scripts' Qwen VAE zeroes `padding` and pads in
its own forward) are handled by reading `original_padding`, so they do not need their own patch; when
their height chunking is actually engaged they keep their own path, because chunking is a memory
decision rather than a speed one.

`SD_SCRIPTS_VAE_CONV_BACKEND=torch` turns the patch off (for A/B runs), `=feather-ops` requires it.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BACKEND_ENV = "SD_SCRIPTS_VAE_CONV_BACKEND"  # auto | feather-ops | torch

_feather_ops_conv2d = None
_feather_ops_checked = False
_backend_logged = False


def _import_feather_conv2d():
    """The `feather_ops.conv2d` module, or None when the extension is not installed (or not on a GPU)."""
    global _feather_ops_conv2d, _feather_ops_checked
    if not _feather_ops_checked:
        _feather_ops_checked = True
        try:
            import feather_ops.conv2d as conv2d_module

            _feather_ops_conv2d = conv2d_module
        except Exception as e:  # ImportError, or a ROCm extension that fails to load
            logger.info(f"feather_ops.conv2d is not available, VAE convs stay on torch: {e}")
    return _feather_ops_conv2d


def feather_conv_available() -> bool:
    return _import_feather_conv2d() is not None and torch.cuda.is_available()


def conv_backend() -> str:
    """`"feather-ops"` or `"torch"`, resolved once from `SD_SCRIPTS_VAE_CONV_BACKEND` (default auto)."""
    global _backend_logged
    requested = os.environ.get(BACKEND_ENV, "auto").strip().lower()
    if requested in ("torch", "none", "off", "0", "false"):
        backend = "torch"
    elif requested in ("auto", "feather-ops", "featherops", "1", "true"):
        backend = "feather-ops" if feather_conv_available() else "torch"
        if requested != "auto" and backend == "torch":
            raise RuntimeError(f"{BACKEND_ENV}={requested} but feather_ops.conv2d is not usable")
    else:
        raise ValueError(f"unknown {BACKEND_ENV}={requested!r} (expected auto, feather-ops or torch)")
    if not _backend_logged:
        _backend_logged = True
        logger.info(f"VAE conv backend: {backend} ({BACKEND_ENV}={requested})")
    return backend


def _sub_padding(module: nn.Module) -> Any:
    """The padding the module's own forward applies inside one conv call.

    `ChunkedConv2d` zeroes `padding` and pads the input itself, keeping the original value in
    `original_padding`; for it (and for a plain `nn.Conv2d`) this is the padding the conv really has.
    """
    return getattr(module, "original_padding", module.padding)


def _is_candidate(module: nn.Module) -> bool:
    """Geometry of one module: 3x3, stride 1, padding 1, dilation 1, no groups, zero padding."""
    if not isinstance(module, nn.Conv2d):
        return False
    if isinstance(module, _FeatherConv2dMixin):
        return True  # already relabelled, so it was accepted once
    if type(module) is not nn.Conv2d and not hasattr(module, "original_padding"):
        # a subclass with its own forward (a masked or a weight-normalized conv); leave it alone
        return False
    return (
        tuple(module.kernel_size) == (3, 3)
        and tuple(module.stride) == (1, 1)
        and tuple(_sub_padding(module)) == (1, 1)
        and tuple(module.dilation) == (1, 1)
        and module.groups == 1
        and module.padding_mode == "zeros"
    )


class _FeatherConv2dMixin:
    """The fast path in front of `super().forward()`, for modules whose geometry `_is_candidate` accepted."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = _try_feather_conv2d(self, input)
        if output is None:
            return super().forward(input)
        return output

    def _apply(self, fn, recurse: bool = True):
        # a dtype/device move invalidates the prepack; drop it (rather than keep a stale copy) and let the next
        # forward repack, which also frees the old copy when the VAE is parked on the CPU
        result = super()._apply(fn, recurse)
        self.__dict__.pop("_feather_conv_packed", None)
        return result


_RELABELLED: Dict[type, type] = {}


def _relabelled(cls: type) -> type:
    """A cached subclass of `cls` that adds the fast path. Same attributes, same `isinstance` checks."""
    if cls not in _RELABELLED:
        _RELABELLED[cls] = type(f"Feather{cls.__name__}", (_FeatherConv2dMixin, cls), {})
    return _RELABELLED[cls]


def _packed_weight(module: nn.Module) -> torch.Tensor:
    """The prepacked weight of a module, keyed on the weight version so a load or a cast repacks once."""
    conv2d_module = _import_feather_conv2d()
    weight = module.weight
    key = conv2d_module.weight_cache_key(weight)
    cached = getattr(module, "_feather_conv_packed", None)
    if cached is None or cached[0] is not weight or cached[1] != key:
        cached = (weight, key, conv2d_module.prepack_conv2d_weight_3x3(weight))
        module._feather_conv_packed = cached
    return cached[2]


def _try_feather_conv2d(module: nn.Module, input: torch.Tensor) -> Optional[torch.Tensor]:
    """The kernel's (or MIOpen's) result, or None to let the module's own forward run."""
    if conv_backend() != "feather-ops" or not isinstance(input, torch.Tensor) or input.ndim != 4:
        return None

    weight = module.weight
    bias = module.bias
    if input.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != input.dtype:
        return None
    if not (input.is_cuda and weight.is_cuda and input.device == weight.device):
        return None
    if not (input.is_contiguous() and weight.is_contiguous()):
        return None
    if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
        return None
    if bias is not None and (bias.dtype != input.dtype or not bias.is_contiguous() or bias.numel() != weight.shape[0]):
        return None

    # a chunked module whose height chunking is engaged keeps its own path: chunking is a memory decision
    chunk_size = getattr(module, "spatial_chunk_size", None)
    if chunk_size is not None and input.shape[2] > chunk_size + module.kernel_size[0] + chunk_size // 4:
        return None

    conv2d_module = _import_feather_conv2d()
    backend = conv2d_module.select_backend(input.dtype, input.shape[1], weight.shape[0], input.shape[2], input.shape[3])
    if backend == "kernel":
        return conv2d_module.conv3x3_prepacked(input, _packed_weight(module), bias)
    if backend == "miopen":
        # MIOpen's Winograd (fp16) or its implicit GEMM (no tile for this shape) wins; the wrapper forces
        # cuDNN/MIOpen on for the call and the heuristic find mode, and touches no global state
        return conv2d_module.conv2d(input, weight, bias, module.stride, _sub_padding(module), module.dilation, module.groups)
    return None


def patch_conv_modules(model: nn.Module, label: str = "vae") -> Dict[str, int]:
    """Relabel the eligible 3x3 stride-1 convs of `model` in place, so encode and decode both use the kernel.

    Relabelling (rather than replacing modules) keeps the parameters, buffers, names and `state_dict` keys
    of the model untouched, so it can happen after the checkpoint is loaded and nothing else has to know.
    Returns a small count for the log; a no-op when the backend is torch.
    """
    stats = {"candidates": 0, "patched": 0}
    if conv_backend() != "feather-ops":
        return stats

    for module in model.modules():
        if not _is_candidate(module):
            continue
        stats["candidates"] += 1
        if isinstance(module, _FeatherConv2dMixin):
            continue
        module.__class__ = _relabelled(type(module))
        stats["patched"] += 1

    if stats["patched"]:
        logger.info(
            f"{label}: {stats['patched']} of {stats['candidates']} 3x3 stride-1 convs use the feather kernel "
            f"(fp16 and untiled shapes delegate to MIOpen)"
        )
    return stats


def is_patched(model: nn.Module) -> bool:
    return any(isinstance(module, _FeatherConv2dMixin) for module in model.modules())
