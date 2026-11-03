"""Deterministic and resumable training order for the training dataloader.

The stock dataloader is `DataLoader(dataset, batch_size=1, shuffle=True, ...)`: the epoch permutation is drawn from
torch's global RNG when the epoch starts, and the position inside the epoch is not recorded anywhere. A `--resume`
therefore has to replay the dataloader and discard batches (`--skip_until_initial_step`), the permutation is redrawn
from a different RNG state, and the RNG stream of the resumed run is offset from the run that was interrupted.

This module provides the sampler used by `--deterministic_dataloader`: the permutation of an epoch is a pure function
of `(seed, epoch)`, so the position inside the epoch can be seeked to. Together with the RNG states that accelerate
saves anyway, that makes a resume continue where the interrupted run stopped: same batches, same noise, no replay.

Two behaviours of the wrapped dataloader that accelerate builds are handled here, because they silently break the naive
version of this:

* `DataLoaderShard.set_epoch()` propagates its own 0-based iteration counter to any sampler that has a `set_epoch`
  method, and `set_epoch` would also reset the position. The sampler therefore has no `set_epoch`; the training loop sets
  the epoch and position itself, and epochs are 0-based (the first epoch is 0), matching accelerate's counter and the
  `epoch` stored in the state.
* `DataLoaderShard.__iter__()` fetches one batch ahead of the one it yields, so a counter inside the sampler runs one
  ahead of what the loop has trained. The sampler's `position` is only informational; the position written to the state
  is the training loop's count of consumed batches, and on resume the position is derived from the saved step count
  (`resolve_state`), which is convention-free.

Exactness needs the dataloader to not read ahead, i.e. `--max_data_loader_n_workers 0`: with workers the sampler is
iterated further ahead by the prefetch depth, and worker RNG states are not part of the checkpoint.
"""

import json
import logging
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

STATE_FILE_NAME = "dataloader_state.json"


class DeterministicRandomSampler(Sampler[int]):
    """Index sampler with a reproducible per-epoch permutation and a seekable position.

    `epoch` is 0-based (the first epoch is 0) and matches both the training loop's epoch index and the counter that
    accelerate propagates to dataloader samplers. The permutation is drawn from a private `torch.Generator`, so sampling
    does not consume the global RNG (which drives noise and timestep sampling).
    """

    def __init__(self, data_source: Union[Dataset, Sequence], seed: int = 0, epoch: int = 0):
        self.num_samples = len(data_source)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.position = 0  # number of indices yielded in this epoch (runs ahead of the trained batches, see the module doc)
        self._permutation_epoch: Optional[int] = None
        self._permutation: List[int] = []

    def set_epoch_position(self, epoch: int, position: int = 0) -> None:
        """Start `epoch` at `position`. Deliberately not called `set_epoch`, see the module docstring."""
        self.epoch = int(epoch)
        self.position = int(position)
        self._permutation_epoch = None

    def set_position(self, position: int) -> None:
        self.position = max(0, min(int(position), self.num_samples))

    def epoch_permutation(self) -> List[int]:
        if self._permutation_epoch != self.epoch:
            generator = torch.Generator()
            generator.manual_seed((self.seed * 1000003 + self.epoch) & 0x7FFF_FFFF_FFFF_FFFF)
            self._permutation = torch.randperm(self.num_samples, generator=generator).tolist()
            self._permutation_epoch = self.epoch
        return self._permutation

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        permutation = self.epoch_permutation()
        for i in range(self.position, self.num_samples):
            self.position = i + 1
            yield permutation[i]

    def state_dict(self) -> Dict[str, Any]:
        return {"epoch": self.epoch, "position": self.position, "seed": self.seed, "num_samples": self.num_samples}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("num_samples") != self.num_samples:
            raise ValueError(f"num_samples mismatch: state {state.get('num_samples')} vs sampler {self.num_samples}")
        if state.get("seed") != self.seed:
            raise ValueError(f"seed mismatch: state {state.get('seed')} vs sampler {self.seed}")
        self.set_epoch_position(state["epoch"], state["position"])


def state_path(directory: str) -> str:
    return os.path.join(directory, STATE_FILE_NAME)


def save_state(directory: str, sampler: DeterministicRandomSampler, epoch: int, position: int, num_processes: int = 1) -> None:
    """Write the dataloader position into a state directory (the same directory `Accelerator.save_state` uses).

    `epoch` is 0-based and `position` is the number of batches the training loop has consumed in that epoch (not the
    sampler's own counter, which runs one batch ahead).
    """
    state = {
        "epoch": int(epoch),
        "position": int(position),
        "seed": sampler.seed,
        "num_samples": sampler.num_samples,
        "num_processes": int(num_processes),
    }
    path = state_path(directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    logger.info(f"save dataloader state to {path}: epoch {state['epoch']} position {state['position']}")


def load_state(directory: str) -> Optional[Dict[str, Any]]:
    """Read the dataloader state of a state directory, or None if the directory has none."""
    path = state_path(directory)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_state(
    state: Dict[str, Any],
    sampler: DeterministicRandomSampler,
    current_step: int,
    current_epoch: int,
    gradient_accumulation_steps: int = 1,
) -> Tuple[int, int]:
    """Seek `sampler` to the batch that follows `current_step` completed steps, and return `(epoch, position)`.

    The position is derived from the step count rather than read from the state file: the file's own counter can be
    one batch ahead (accelerate's dataloader fetches one batch ahead of the one it yields) and older files used other
    conventions. `current_epoch` is the 1-based epoch from `train_state.json`; the sampler's epoch is 0-based.
    """
    num_samples = sampler.num_samples
    if num_samples <= 0:
        raise ValueError("cannot resume with an empty dataset")

    # a changed dataset (added or removed images, `max_train_images` picking a different subset, ...) cannot reproduce the
    # batch order of the interrupted run, but the run should still continue at the saved step instead of restarting
    if state.get("num_samples") != num_samples:
        logger.warning(
            f"dataloader state was saved for {state.get('num_samples')} batch indices, the current dataset has {num_samples}:"
            f" a dataset whose length changed (added or removed images, `max_train_images` re-sampling, images dropped while"
            f" caching latents, ...) cannot reproduce the batch order of the remaining epochs, so epoch-derived step totals may"
            f" have changed too. Continuing at the saved step"
        )
    if state.get("seed") != sampler.seed:
        logger.warning(
            f"dataloader state was saved with seed {state.get('seed')}, this run uses seed {sampler.seed}: the batch order"
            f" cannot match the interrupted run, continuing at the saved step"
        )

    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

    # completed steps -> batches consumed, minus the epochs that are already finished
    steps_per_epoch = max(1, math.ceil(num_samples / gradient_accumulation_steps))
    epoch = int(current_epoch) - 1  # 0-based
    position = int(current_step) * gradient_accumulation_steps - epoch * steps_per_epoch * gradient_accumulation_steps
    position = max(0, min(position, num_samples))
    if position >= num_samples:  # the state was saved at an epoch boundary
        epoch += 1
        position = 0

    sampler.set_epoch_position(epoch, position)
    return epoch, position
