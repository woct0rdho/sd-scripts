"""Adaptive partial gradient checkpointing: pick how many blocks keep their activations, from the token count.

Partial gradient checkpointing (`enable_gradient_checkpointing(skip_blocks=n)` in the MOSS-Image and Anima models)
checkpoints the first `num_blocks - n` blocks and lets the last `n` keep their activations. The peak is linear in both
the number of uncheckpointed blocks and the sequence length, so

    peak_gib(tokens, n) = fixed_gib + tokens / 1024 * (per_1k_tokens_gib + n * per_block_1k_tokens_gib)

`fixed_gib` is *measured* at the first step (`torch.cuda.memory_allocated()`): weights, LoRA parameters and their
gradients, optimizer state and its workspace, whatever else the run keeps resident. Measuring it is what makes one set
of coefficients work on CUDA and on ROCm, for Anima and for MOSS-Image, and across optimizer versions - those differ by
1-2 GiB of resident buffers that no closed form predicts.

The two coefficients are the activations, and they follow from the block source (see `architecture_coefficients`).
They do not depend on the GPU, the attention kernel or the allocator, and the measurements in the model guides agree
with the derivation to within 25% on every combination tried (Anima and MOSS-Image, CUDA and ROCm).

The one term the coefficients do not model is the caching allocator's reserved-minus-allocated slack, so the model
predicts the *allocated* peak and `--gpu_memory_reserved_gib` covers that slack plus anything else that starts using the
GPU after the first step. `allocator_hint` says how to keep the slack small; the guides have the numbers.

The decision is a pure Python flag flip in `Block.forward`. `torch.compile` compiles `block._forward` - outside the
checkpointing wrapper - so changing `n` between steps does not invalidate or add a single compiled graph. (That is why
`--torch_compile`, which compiles the whole model, is not supported: there the checkpointing branch is inside the
compiled graph.)
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

# Measured `(per_1k_tokens_gib, per_block_1k_tokens_gib)` per architecture; the model guides have the measurements and
# the derivation. Both backends give the same coefficients (the activation bytes are a property of the block), so one
# entry serves Windows/CUDA and Linux/ROCm.
MEMORY_MODELS: dict[str, tuple[float, float]] = {
    "MossImage": (0.428, 0.105),
    "Anima": (0.243, 0.078),
}

# `architecture_coefficients` is within a factor 1.21 of those measurements (Moss's per-block term, whose RMSNorm keeps
# an fp32 copy of the norm output); the factor makes the derived coefficients over-predict rather than under-predict.
ARCH_SAFETY = 1.15

DEFAULT_RESERVED_GIB = 4.0

# Enough of `--gpu_memory_reserved_gib` when the allocator is not fragmenting; see `allocator_hint` and the guides.
EXPANDABLE_RESERVED_GIB = 1.5


@dataclass(frozen=True)
class MemoryModel:
    """`peak_gib(tokens, skip_blocks)` and its inverse for one architecture."""

    fixed_gib: float
    per_1k_tokens_gib: float
    per_block_1k_tokens_gib: float
    source: str = ""

    def peak_gib(self, tokens: int, skip_blocks: int) -> float:
        return self.fixed_gib + tokens / 1024.0 * (self.per_1k_tokens_gib + skip_blocks * self.per_block_1k_tokens_gib)

    def max_skip_blocks(self, tokens: int, budget_gib: float, num_blocks: int) -> int:
        """Largest `n` whose predicted peak fits `budget_gib`, clamped to `[0, num_blocks]`."""
        if tokens <= 0:
            return num_blocks
        room = budget_gib - self.fixed_gib - tokens / 1024.0 * self.per_1k_tokens_gib
        if room <= 0:
            return 0
        n = int(room / (tokens / 1024.0 * self.per_block_1k_tokens_gib))
        return max(0, min(n, num_blocks))


def parameter_gib(model: torch.nn.Module) -> float:
    """Bytes of the resident parameters (weights) and of the trainable ones with their gradients, in GiB."""
    weight_bytes = 0
    trainable_bytes = 0
    for param in model.parameters():
        nbytes = param.numel() * param.element_size()
        weight_bytes += nbytes
        if param.requires_grad:
            trainable_bytes += nbytes
    return (weight_bytes + 2 * trainable_bytes) / 2**30  # grad in the same dtype as the parameter


def allocator_hint() -> Optional[str]:
    """A one-time warning when the allocator is the fragmenting default, or None.

    The coefficients predict the *allocated* peak; the caching allocator's reserved-minus-allocated slack is what
    `--gpu_memory_reserved_gib` has to cover, and the default allocator keeps it several times larger than
    `expandable_segments:True` does (the model guides have the numbers). Nothing is changed at runtime - the setting has
    to be in the environment before the allocator is created.
    """
    if not torch.cuda.is_available():
        return None
    if _expandable_segments_enabled():
        return None
    return (
        "the default caching allocator keeps the step's peak well above the live tensors (both backends), so"
        " --gpu_memory_reserved_gib has to cover it; PYTORCH_ALLOC_CONF=expandable_segments:True shrinks it several times"
        f" over, and {EXPANDABLE_RESERVED_GIB} GiB of reserve is enough there"
    )


def _expandable_segments_enabled() -> bool:
    """Whether the caching allocator runs with expandable segments.

    `PYTORCH_ALLOC_CONF` is the current name; the legacy `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_HIP_ALLOC_CONF` still work
    but are parsed *first*, so a legacy setting overrides the new one. The allocator's own report is authoritative (it
    also sees a setting changed at runtime), and the environment is the fallback for a torch without it.
    """
    try:
        return bool(torch.cuda.memory._snapshot()["allocator_settings"].get("expandable_segments", False))
    except Exception:  # no snapshot API, or the allocator refused to snapshot
        pass
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        value = os.environ.get(var)
        if value is not None:  # the first one that is set is the one torch parses
            return "expandable_segments:true" in value.lower().replace(" ", "")
    return False


def architecture_coefficients(model: torch.nn.Module) -> Optional[tuple[float, float]]:
    """`(per_1k_tokens_gib, per_block_1k_tokens_gib)` from the block geometry, or None if it is not known.

    A block that keeps its activations holds, per token, `q, k, v`, the attention output, both norm outputs,
    `ffn_up`/`ffn_gate` and the SiTU-GLU product in bf16, plus the INT8 codes its four linears' dgrad consumes:
    `11*dim + 7*ffn` bytes. A checkpointed block holds two hidden-sized tensors: `4*dim` bytes per token per block. Both
    are scaled by `ARCH_SAFETY` because the derivation is within a factor 1.25 of the measurements (a block whose norms
    keep fp32 copies, like MOSS-Image's, is above the formula).
    """
    dim = getattr(model, "dim", None) or getattr(model, "hidden_size", None) or getattr(model, "model_channels", None)
    ffn = getattr(model, "ffn_dim", None) or getattr(model, "ffn_hidden_size", None)
    num_blocks = getattr(model, "num_layers", None) or getattr(model, "num_blocks", None)
    if not all(isinstance(v, int) and v > 0 for v in (dim, ffn, num_blocks)):
        return None
    per_1k = num_blocks * 4 * dim / 2**20
    per_block = (11 * dim + 7 * ffn) / 2**20
    return per_1k * ARCH_SAFETY, per_block * ARCH_SAFETY


def resolve_memory_model(args, model: torch.nn.Module, allocated_gib: Optional[float] = None) -> MemoryModel:
    """The coefficients for this run: user overrides win, then the measured table, then the block geometry.

    `allocated_gib` is the resident part measured at the first step; without it `parameter_gib` stands in, which is only
    for tests and dry runs (it cannot see the optimizer state).
    """
    name = type(model).__name__
    fixed = getattr(args, "adaptive_checkpointing_fixed", None)
    per_1k = getattr(args, "adaptive_checkpointing_per_1k_tokens", None)
    per_block = getattr(args, "adaptive_checkpointing_per_block_1k_tokens", None)

    if per_1k is None or per_block is None:
        default = MEMORY_MODELS.get(name) or architecture_coefficients(model)
        if default is None:
            raise ValueError(
                f"--adaptive_checkpointing cannot derive a memory model for {name}: it has no `dim`/`ffn_dim`/`num_layers`,"
                f" so pass --adaptive_checkpointing_per_1k_tokens and --adaptive_checkpointing_per_block_1k_tokens"
                f" (GiB per 1024 tokens, with and without one uncheckpointed block)."
                f" Measured: {', '.join(sorted(MEMORY_MODELS)) or 'none'}"
            )
        default_per_1k, default_per_block = default
        per_1k = default_per_1k if per_1k is None else per_1k
        per_block = default_per_block if per_block is None else per_block
        if (per_1k, per_block) != default:
            source = "user coefficients"
        elif name in MEMORY_MODELS:
            source = f"measured ({name})"
        else:
            source = f"derived from {name}'s block geometry"
    else:
        source = "user coefficients"

    if fixed is None:
        if allocated_gib is not None:
            fixed = allocated_gib
            source += ", fixed measured"
        else:
            # tests and dry runs only: the weights and their gradients, without the optimizer state the real
            # measurement at the first step always includes
            fixed = parameter_gib(model)
            source += ", fixed from parameters"
    else:
        source += ", fixed from --adaptive_checkpointing_fixed"

    if per_block <= 0:
        raise ValueError(f"--adaptive_checkpointing_per_block_1k_tokens must be positive, got {per_block}")
    return MemoryModel(
        fixed_gib=fixed,
        per_1k_tokens_gib=per_1k,
        per_block_1k_tokens_gib=per_block,
        source=source,
    )


def token_count(latents: torch.Tensor, patch_size: int = 2) -> int:
    """Sequence length of the image part: `B * (H / patch) * (W / patch)` for VAE latents `[B, C, H, W]`."""
    if latents.ndim == 5:  # [B, C, 1, H, W] caches
        latents = latents.squeeze(2)
    batch, _, height, width = latents.shape
    return int(batch) * (int(height) // patch_size) * (int(width) // patch_size)


def total_gpu_gib(device: Optional[torch.device] = None) -> Optional[float]:
    """Total memory of the device in GiB (None when there is no GPU); only used for the log."""
    if not torch.cuda.is_available():
        return None
    try:
        _, total = torch.cuda.mem_get_info(device)
    except Exception:
        return None
    return total / 2**30


def free_gpu_gib(device: Optional[torch.device] = None) -> Optional[float]:
    """Memory this process can still allocate before the driver refuses, in GiB (None when there is no GPU).

    That is the driver's free memory plus the part of the caching allocator's reserved pool that is not in use: the
    loader leaves sizeable cached blocks behind, and they are available to the step. It stays below the driver's limit
    because the reserved pool is already taken from the driver.
    """
    if not torch.cuda.is_available():
        return None
    try:
        driver_free, _ = torch.cuda.mem_get_info(device)
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    except Exception:  # an unsupported device type or a driver without mem_get_info
        return None
    return (driver_free + max(cached, 0)) / 2**30


class AdaptiveCheckpointing:
    """Chooses `skip_blocks` from the batch's token count and applies it when it changes.

    The budget is `--adaptive_checkpointing_budget` when given, else what is allocatable at the first step plus the
    resident part minus `--gpu_memory_reserved_gib`. Measuring at the first step is what makes it simple: by then the
    DiT, the LoRA, the text encoder and the VAE are wherever the run keeps them, so what is still free is what the step
    may use. The reserve covers what the model does not (the allocator's slack, anything else that grows later).
    """

    def __init__(
        self,
        memory_model: MemoryModel,
        num_blocks: int,
        budget_gib: float,
        apply: Callable[[int], None],
        log_limit: int = 10,
    ):
        self.memory_model = memory_model
        self.num_blocks = num_blocks
        self.budget_gib = budget_gib
        self.apply = apply
        self.log_limit = log_limit
        self.skip_blocks: Optional[int] = None
        self._logged = 0
        self._logged_tokens: set[tuple[int, int]] = set()

    @classmethod
    def from_args(cls, args, model: torch.nn.Module, apply: Callable[[int], None]) -> "AdaptiveCheckpointing":
        # measured where it matters: whatever is resident at the first step (weights, LoRA, grads, optimizer state and its
        # workspace) is what the model's `fixed_gib` has to be, and no closed form predicts it across optimizers
        allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else None
        memory_model = resolve_memory_model(args, model, allocated)
        num_blocks = getattr(model, "num_layers", None) or getattr(model, "num_blocks", None)
        if not isinstance(num_blocks, int) or num_blocks <= 0:
            raise ValueError(f"--adaptive_checkpointing needs a block count on the model, {type(model).__name__} has none")
        budget = getattr(args, "adaptive_checkpointing_budget", None)
        if budget is None:
            free = free_gpu_gib()
            if free is None:
                raise ValueError(
                    "--adaptive_checkpointing cannot read the free GPU memory; pass --adaptive_checkpointing_budget"
                )
            reserved = getattr(args, "gpu_memory_reserved_gib", DEFAULT_RESERVED_GIB)
            # `free` is what is left *after* everything resident, so adding the resident part back turns it into a ceiling
            # for `peak_gib`, which counts the resident part; the same config then works on a 16 GiB card and on a 124 GiB
            # one (the free memory carries the card size)
            budget = free + memory_model.fixed_gib - reserved
            total = total_gpu_gib()
            logger.info(
                f"adaptive checkpointing: {free:.1f} GiB allocatable"
                + (f" of {total:.1f} GiB" if total is not None else "")
                + f" (driver free + own cache), {memory_model.fixed_gib:.2f} GiB of it resident"
                f" -> budget {budget:.1f} GiB (reserved {reserved:.1f} GiB)"
            )
        if budget <= memory_model.fixed_gib:
            raise ValueError(
                f"--adaptive_checkpointing budget {budget:.1f} GiB is below the {memory_model.fixed_gib:.1f} GiB that is "
                f"resident before the step; lower --gpu_memory_reserved_gib or pass --adaptive_checkpointing_budget"
            )
        logger.info(
            f"adaptive checkpointing: budget {budget:.1f} GiB, fixed {memory_model.fixed_gib:.2f} GiB (resident), "
            f"{memory_model.per_1k_tokens_gib:.3f} GiB per 1024 tokens + {memory_model.per_block_1k_tokens_gib:.3f} GiB "
            f"per 1024 tokens per uncheckpointed block ({memory_model.source}), {num_blocks} blocks"
        )
        hint = allocator_hint()
        if hint is not None:
            logger.warning(f"adaptive checkpointing: {hint}")
        return cls(memory_model, num_blocks, budget, apply)

    def skip_blocks_for(self, tokens: int) -> int:
        return self.memory_model.max_skip_blocks(tokens, self.budget_gib, self.num_blocks)

    def update(self, tokens: Optional[int]) -> Optional[int]:
        """Apply the `skip_blocks` for this batch's token count; returns it (None when the count is unknown)."""
        if not tokens:
            return self.skip_blocks
        n = self.skip_blocks_for(tokens)
        if n == 0 and (tokens, -1) not in self._logged_tokens:
            worst = self.memory_model.peak_gib(tokens, 0)
            if worst > self.budget_gib:
                self._logged_tokens.add((tokens, -1))
                logger.warning(
                    f"adaptive checkpointing: {tokens} tokens need {worst:.1f} GiB even with every block checkpointed, "
                    f"more than the {self.budget_gib:.1f} GiB budget; this batch will probably run out of memory. "
                    f"Lower --gpu_memory_reserved_gib, train a smaller resolution, or use --unsloth_offload_checkpointing"
                )
        if n == self.skip_blocks:
            return n
        previous = self.skip_blocks
        self.skip_blocks = n
        self.apply(n)

        # one line per distinct (tokens, n) so a bucketed run does not spam the log, and DEBUG afterwards
        key = (int(tokens), n)
        message = (
            f"adaptive checkpointing: {tokens} tokens -> {self.num_blocks - n}/{self.num_blocks} blocks checkpointed "
            f"(skip {n}), predicted peak {self.memory_model.peak_gib(tokens, n):.1f} GiB of {self.budget_gib:.1f} GiB"
        )
        if previous is not None:
            message += f", was skip {previous}"
        if key not in self._logged_tokens and self._logged < self.log_limit:
            self._logged_tokens.add(key)
            self._logged += 1
            logger.info(message)
        else:
            logger.debug(message)
        return n
