"""Adaptive partial gradient checkpointing: pick how many blocks keep their activations, from the token count.

Partial gradient checkpointing (`enable_gradient_checkpointing(skip_blocks=n)` in the MOSS-Image and Anima models)
checkpoints the first `num_blocks - n` blocks and lets the last `n` keep their activations. The memory cost is linear
in both the number of uncheckpointed blocks and the sequence length, so for one architecture it is

    peak_gib(tokens, n) = fixed_gib + tokens / 1024 * (per_1k_tokens_gib + n * per_block_1k_tokens_gib)

`fixed_gib` is the resident part (weights, LoRA, their gradients); the two coefficients are per architecture and were
measured on gfx1151 with the ConvRot INT8 MOSS-Image DiT and a rank-64 LoRA (see the constants below). The same shape of
model is what makes the fit accurate: the retained bytes of a block are its attention (q/k/v and their norms), its
gated FFN intermediates (2x ffn up/gate, the SiTU-GLU product, the down input) and the inputs of the LoRA branches.

Adaptive mode uses the *measured* coefficients as defaults, because a wrong coefficient means either an OOM or a needless
recompute, and calibrating them at runtime (two extra forward+backward passes with different `n`) would cost seconds,
risk an OOM at the largest bucket, and add compile variants. `fixed_gib` is cheap to get right, so it is computed from
the loaded parameters unless the user overrides it. Every term can be overridden per run, which is also how another
architecture (or a different LoRA rank) is supported.

The decision is a pure Python flag flip in `Block.forward`. `torch.compile` compiles `block._forward` - outside the
checkpointing wrapper - so changing `n` between steps does not invalidate or add a single compiled graph. (That is why
`--torch_compile`, which compiles the whole model, is not supported: there the checkpointing branch is inside the
compiled graph.)
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

# Per-architecture coefficients of `peak_gib(tokens, n)` above, measured on gfx1151 at 4096/9216/16384 tokens
# (`~/tmp/sd-scripts/moss_step_memory.py`, int8 ConvRot DiT + rank-64 LoRA, eager): 39.05 GiB at 4096 tokens with all
# blocks uncheckpointed, 13.69 GiB at 16384 with all checkpointed, 71.07 GiB at 16384 with 20 uncheckpointed. A block's
# retained activations are 179 KB per token (2.87 GiB at 16384 tokens); the same fit is within 1.5 GiB of every point.
#
# (architecture class name -> (per_1k_tokens_gib, per_block_1k_tokens_gib))
MEMORY_MODELS = {
    "MossImage": (0.50, 0.179),
}

# `fixed_gib` is the token-independent part: the parameters, the trainable parameters with their gradients, and the
# allocator overhead that the fit sees but `parameter_gib` cannot count (blocks reserved for the previous step's buffers,
# rounding). Measured as a 2 GiB residual on this machine; predicting high is the safe direction.
ALLOCATOR_OVERHEAD_GIB = 2.0

DEFAULT_RESERVED_GIB = 4.0


@dataclass(frozen=True)
class MemoryModel:
    """`peak_gib(tokens, skip_blocks)` and its inverse for one architecture."""

    fixed_gib: float
    per_1k_tokens_gib: float
    per_block_1k_tokens_gib: float
    source: str = ""
    resident_gib: float = 0.0  # part of `fixed_gib` already on the GPU when the budget is measured

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


def resolve_memory_model(args, model: torch.nn.Module) -> MemoryModel:
    """The coefficients for this run: user overrides win, then the measured table, then an error."""
    name = type(model).__name__
    fixed = getattr(args, "adaptive_checkpointing_fixed", None)
    per_1k = getattr(args, "adaptive_checkpointing_per_1k_tokens", None)
    per_block = getattr(args, "adaptive_checkpointing_per_block_1k_tokens", None)

    if per_1k is None or per_block is None:
        if name not in MEMORY_MODELS:
            raise ValueError(
                f"--adaptive_checkpointing has no measured memory model for {name}: pass "
                f"--adaptive_checkpointing_per_1k_tokens and --adaptive_checkpointing_per_block_1k_tokens "
                f"(GiB per 1024 tokens, with and without one uncheckpointed block). "
                f"Known: {', '.join(sorted(MEMORY_MODELS))}"
            )
        default_per_1k, default_per_block = MEMORY_MODELS[name]
        per_1k = default_per_1k if per_1k is None else per_1k
        per_block = default_per_block if per_block is None else per_block
        source = f"measured ({name})" if (per_1k, per_block) == (default_per_1k, default_per_block) else "user coefficients"
    else:
        source = "user coefficients"

    if fixed is None:
        fixed = parameter_gib(model) + ALLOCATOR_OVERHEAD_GIB
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
        resident_gib=parameter_gib(model),
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

    The budget is `--adaptive_checkpointing_budget` when given, else the free GPU memory at the first step minus
    `--gpu_memory_reserved_gib`. Measuring at the first step is what makes the reserve simple: by then the DiT, the
    LoRA, the text encoder and the VAE are wherever the run keeps them, so whatever is still free is what the step may
    use. The reserve covers the allocator's fragmentation, the optimizer state and anything the user knows about.
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
        memory_model = resolve_memory_model(args, model)
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
            # `peak_gib` counts the weights, so add back what is already resident; this is what makes the same config
            # work on a 32 GiB card and on a 124 GiB one (the free memory carries the card size)
            budget = free + memory_model.resident_gib - reserved
            total = total_gpu_gib()
            logger.info(
                f"adaptive checkpointing: {free:.1f} GiB allocatable"
                + (f" of {total:.1f} GiB" if total is not None else "")
                + f" (driver free + own cache), {memory_model.resident_gib:.1f} GiB of it resident parameters"
                f" -> budget {budget:.1f} GiB (reserved {reserved:.1f} GiB)"
            )
        if budget <= memory_model.fixed_gib:
            raise ValueError(
                f"--adaptive_checkpointing budget {budget:.1f} GiB is below the {memory_model.fixed_gib:.1f} GiB the "
                f"parameters alone need; lower --gpu_memory_reserved_gib or pass --adaptive_checkpointing_budget"
            )
        logger.info(
            f"adaptive checkpointing: budget {budget:.1f} GiB, fixed {memory_model.fixed_gib:.1f} GiB, "
            f"{memory_model.per_1k_tokens_gib:.3f} GiB per 1024 tokens + {memory_model.per_block_1k_tokens_gib:.3f} GiB "
            f"per 1024 tokens per uncheckpointed block ({memory_model.source}), {num_blocks} blocks"
        )
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
