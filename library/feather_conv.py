"""Feather ops 3x3 stride-1 conv2d for the VAE, as a drop-in that keeps torch's fallbacks.

`feather_ops` runs a HIP kernel for bf16 NCHW 3x3 stride-1 padding-1 convolutions and delegates two
cases to MIOpen, where the kernel loses: fp16 (MIOpen's Winograd does 4/9 of the multiplies) and the
shapes the kernel cannot tile (a VAE's first and last conv). Everything else falls back to `F.conv2d`.

VAEs are used in two places: encoding the training images (latent caching, `--cache_latents`) and
decoding the sample images (`--sample_every_n_steps`). Both go through the same model object, so the
wiring is a one-time relabel of its eligible `nn.Conv2d` modules (`patch_conv_modules`), which keeps the
parameters, buffers, module names and `state_dict` keys untouched - the forward of those modules only
gains a fast path in front of `super().forward()`.

The decision and the prepack are feather_ops' (`conv3x3_module_reason`, `conv3x3_eligibility`,
`select_backend`, `prepacked_conv3x3`, `relabel_modules`); this module adds sd-scripts' part: the
backend switch below, the manual-padding/chunking convention of sd-scripts' Qwen VAE (`ChunkedConv2d`
zeroes `padding`, pads the input in its own forward and keeps `original_padding`), and the policy that
the VAE is only patched when the user asked for it.

`SD_SCRIPTS_VAE_CONV_BACKEND=torch` turns the patch off (for A/B runs), `=feather-ops` requires it.
"""

import functools
import logging
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BACKEND_ENV = "SD_SCRIPTS_VAE_CONV_BACKEND"  # auto | feather-ops | torch


def _import_feather_ops():
    """The feather_ops helpers this module uses, or None when the extension is not installed or not a ROCm build."""
    if not getattr(torch.version, "hip", None):
        return None
    try:
        from feather_ops.conv2d import conv2d_miopen, conv3x3_eligibility, conv3x3_prepacked, select_backend
        from feather_ops.conv2d_module import ModuleCache, conv3x3_module_reason, prepacked_conv3x3
        from feather_ops.modules import relabel_modules
    except Exception as e:  # not installed, or the HIP extension failed to load
        logger.info(f"feather_ops is not available, VAE convs stay on torch: {e}")
        return None
    return SimpleNamespace(
        conv2d_miopen=conv2d_miopen,
        conv3x3_eligibility=conv3x3_eligibility,
        conv3x3_prepacked=conv3x3_prepacked,
        conv3x3_module_reason=conv3x3_module_reason,
        prepacked_conv3x3=prepacked_conv3x3,
        relabel_modules=relabel_modules,
        select_backend=select_backend,
        cache=ModuleCache(),
    )


_feather_ops = _import_feather_ops()


def feather_conv_available() -> bool:
    return _feather_ops is not None and torch.cuda.is_available()


@functools.lru_cache(maxsize=1)
def conv_backend() -> str:
    """`"feather-ops"` or `"torch"`, from `SD_SCRIPTS_VAE_CONV_BACKEND` (default auto), decided on first use."""
    requested = os.environ.get(BACKEND_ENV, "auto").strip().lower()
    if requested in ("torch", "none", "off", "0", "false"):
        backend = "torch"
    elif requested in ("auto", "feather-ops", "featherops", "1", "true"):
        backend = "feather-ops" if feather_conv_available() else "torch"
        if requested != "auto" and backend == "torch":
            raise RuntimeError(f"{BACKEND_ENV}={requested} but feather_ops is not usable")
    else:
        raise ValueError(f"unknown {BACKEND_ENV}={requested!r} (expected auto, feather-ops or torch)")
    logger.info(f"VAE conv backend: {backend} ({BACKEND_ENV}={requested})")
    return backend


def _sub_padding(module: nn.Module):
    """The padding the module's own forward applies inside one conv call.

    `ChunkedConv2d` zeroes `padding` and pads the input itself, keeping the original value in
    `original_padding`; for it (and for a plain `nn.Conv2d`) this is the padding the conv really has.
    """
    return getattr(module, "original_padding", module.padding)


def _is_candidate(module: nn.Module) -> bool:
    """Geometry of one module, plus the two shapes of module sd-scripts allows.

    A plain `nn.Conv2d`, or a subclass that only moves the padding into its own forward
    (`original_padding`); a subclass with a forward of its own is left alone.
    """
    if not isinstance(module, nn.Conv2d):
        return False
    if type(module) is not nn.Conv2d and not hasattr(module, "original_padding"):
        return False
    return _feather_ops is not None and _feather_ops.conv3x3_module_reason(module, _sub_padding(module)) is None


class _FeatherConv2dMixin:
    """The fast path in front of `super().forward()`, for modules whose geometry `_is_candidate` accepted."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = _try_feather_conv2d(self, input)
        if output is None:
            return super().forward(input)  # the module's own path: padding, chunking, torch
        return output

    def _apply(self, fn, recurse: bool = True):
        # a dtype/device move invalidates the prepack; drop it now rather than keep a stale copy
        result = super()._apply(fn, recurse)
        if _feather_ops is not None:
            _feather_ops.cache.invalidate(self)
        return result


def _try_feather_conv2d(module: nn.Module, input: torch.Tensor):
    """The kernel's (or MIOpen's) result, or None to let the module's own forward run."""
    if not isinstance(input, torch.Tensor) or input.ndim != 4:
        return None
    if _feather_ops is None or conv_backend() != "feather-ops":
        return None

    # a chunked module whose height chunking is engaged keeps its own path: chunking is a memory decision
    chunk_size = getattr(module, "spatial_chunk_size", None)
    if chunk_size is not None and input.shape[2] > chunk_size + module.kernel_size[0] + chunk_size // 4:
        return None

    padding = _sub_padding(module)
    if (
        _feather_ops.conv3x3_eligibility(
            input, module.weight, module.bias, module.stride, padding, module.dilation, module.groups
        )
        is not None
    ):
        return None

    backend = _feather_ops.select_backend(input.dtype, input.shape[1], module.weight.shape[0], input.shape[2], input.shape[3])
    if backend == "kernel":
        packed, bias = _feather_ops.prepacked_conv3x3(module)
        return _feather_ops.conv3x3_prepacked(input, packed, bias)
    if backend == "miopen":
        # MIOpen's Winograd (fp16) or its implicit GEMM (no tile for this shape) wins; the wrapper forces
        # cuDNN/MIOpen on for the call with the heuristic find mode, and touches no global state
        return _feather_ops.conv2d_miopen(
            input, module.weight, module.bias, module.stride, padding, module.dilation, module.groups
        )
    return None


def patch_conv_modules(model: nn.Module, label: str = "vae") -> int:
    """Relabel the eligible 3x3 stride-1 convs of `model` in place, so encode and decode both use the kernel.

    Relabelling (rather than replacing modules) keeps the parameters, buffers, names and `state_dict` keys
    of the model untouched, so it can happen after the checkpoint is loaded and nothing else has to know.
    Returns the number of convs relabelled; a no-op when the backend is torch.
    """
    if conv_backend() != "feather-ops":
        return 0
    patched = _feather_ops.relabel_modules(model, _FeatherConv2dMixin, _is_candidate)
    if patched:
        logger.info(
            f"{label}: {patched} 3x3 stride-1 convs use the feather kernel (fp16 and untiled shapes delegate to MIOpen)"
        )
    return patched


def is_patched(model: nn.Module) -> bool:
    return any(isinstance(module, _FeatherConv2dMixin) for module in model.modules())
