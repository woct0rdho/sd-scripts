"""Tests for library/resumable_dataloader.py (the `--deterministic_dataloader` sampler).

The point of the module is that a `--resume` continues the *same* run: the batches, the python RNG (caption shuffling)
and the torch RNG (noise, timesteps) after the resume point must be the ones the uninterrupted run would have used. The
integration tests at the bottom check exactly that, with a dataset that records the RNG values it observes.

The tests also pin the two dataloader behaviours that broke the naive version, both of which are properties of
accelerate's `DataLoaderShard` rather than of this module:

* it calls `set_epoch` on any sampler that has one, with its own 0-based counter, so our sampler must not have that
  method and the epoch is 0-based;
* it fetches one batch ahead of the batch it yields, so the sampler's own counter is one ahead of the trained batches and
  the state must not be read back from it.
"""

import json
import random

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from library.resumable_dataloader import (
    STATE_FILE_NAME,
    DeterministicRandomSampler,
    load_state,
    resolve_state,
    save_state,
    state_path,
)


def make_sampler(n=7, seed=42, epoch=0, position=0):
    sampler = DeterministicRandomSampler(range(n), seed=seed, epoch=epoch)
    sampler.set_position(position)
    return sampler


def permutation(sampler):
    return sampler.epoch_permutation()


# region sampler


def test_permutation_is_a_pure_function_of_seed_and_epoch():
    assert permutation(make_sampler(seed=42, epoch=0)) == permutation(make_sampler(seed=42, epoch=0))
    assert permutation(make_sampler(seed=42, epoch=0)) != permutation(make_sampler(seed=42, epoch=1))
    assert permutation(make_sampler(seed=42, epoch=0)) != permutation(make_sampler(seed=43, epoch=0))

    # seed + epoch would collide for (0, 5) and (5, 0); the sampler mixes them
    assert permutation(make_sampler(seed=0, epoch=5)) != permutation(make_sampler(seed=5, epoch=0))


def test_permutation_is_a_permutation():
    indices = permutation(make_sampler(n=64, seed=3, epoch=2))
    assert sorted(indices) == list(range(64))


def test_permutation_does_not_consume_the_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    make_sampler(seed=42, epoch=3).epoch_permutation()
    assert torch.equal(torch.get_rng_state(), before)


def test_sampler_has_no_set_epoch_because_accelerate_would_call_it():
    """`DataLoaderShard.set_epoch` forwards its own 0-based counter to `sampler.set_epoch` on every iteration."""

    sampler = make_sampler()
    assert not hasattr(sampler, "set_epoch")
    assert hasattr(sampler, "set_epoch_position")


def test_position_advances_with_each_consumed_index():
    sampler = make_sampler(n=5)
    iterator = iter(sampler)
    assert sampler.position == 0
    next(iterator)
    assert sampler.position == 1
    for _ in iterator:
        pass
    assert sampler.position == 5


def test_iterating_from_a_position_yields_the_rest_of_the_epoch():
    full = permutation(make_sampler(n=9, seed=42, epoch=0))
    sampler = make_sampler(n=9, seed=42, epoch=0, position=4)
    assert list(sampler) == full[4:]
    assert sampler.position == 9


def test_set_epoch_position_resets_the_position_and_reuses_the_permutation_cache():
    sampler = make_sampler(n=5, seed=42, epoch=0)
    list(sampler)
    assert sampler.position == 5
    sampler.set_epoch_position(1)
    assert sampler.position == 0
    assert list(sampler) == permutation(make_sampler(n=5, seed=42, epoch=1))
    sampler.set_epoch_position(1, 3)  # explicit position wins over the reset
    assert sampler.position == 3


def test_set_position_is_clamped_to_the_epoch():
    sampler = make_sampler(n=5)
    sampler.set_position(-3)
    assert sampler.position == 0
    sampler.set_position(99)
    assert sampler.position == 5


def test_len_is_the_number_of_batch_indices():
    assert len(make_sampler(n=13)) == 13


def test_state_dict_round_trip_reproduces_the_position():
    sampler = make_sampler(n=6, seed=42, epoch=3, position=2)
    state = sampler.state_dict()
    assert state == {"epoch": 3, "position": 2, "seed": 42, "num_samples": 6}

    restored = make_sampler(n=6, seed=42, epoch=0)
    restored.load_state_dict(state)
    assert (restored.epoch, restored.position) == (3, 2)
    assert list(restored) == list(sampler)


def test_load_state_dict_rejects_a_different_dataset_or_seed():
    sampler = make_sampler(n=6, seed=42)
    with pytest.raises(ValueError, match="num_samples"):
        sampler.load_state_dict({"epoch": 0, "position": 0, "seed": 42, "num_samples": 5})
    with pytest.raises(ValueError, match="seed"):
        sampler.load_state_dict({"epoch": 0, "position": 0, "seed": 7, "num_samples": 6})


# endregion


# region state files


def test_state_file_round_trip(tmp_path):
    sampler = make_sampler(n=8, seed=42, epoch=2)
    save_state(str(tmp_path), sampler, epoch=2, position=3, num_processes=1)

    path = state_path(str(tmp_path))
    assert path == str(tmp_path / STATE_FILE_NAME)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw == {"epoch": 2, "position": 3, "seed": 42, "num_samples": 8, "num_processes": 1}
    assert load_state(str(tmp_path)) == raw


def test_load_state_returns_none_without_a_state_file(tmp_path):
    assert load_state(str(tmp_path)) is None


def test_load_state_ignores_fields_it_does_not_know(tmp_path):
    """State files written by earlier versions carry extra keys (e.g. a `format_version`); they must still load."""

    with open(state_path(str(tmp_path)), "w", encoding="utf-8") as f:
        json.dump({"format_version": 1, "epoch": 0, "position": 1601, "seed": 42, "num_samples": 8}, f)
    state = load_state(str(tmp_path))
    assert state["num_samples"] == 8 and state["position"] == 1601


# endregion


# region resolving the position from the step count


def test_resolve_state_derives_the_position_from_the_step_count():
    sampler = make_sampler(n=8, seed=42)
    state = {"epoch": 0, "position": 0, "seed": 42, "num_samples": 8}
    assert resolve_state(state, sampler, current_step=3, current_epoch=1) == (0, 3)
    assert list(sampler) == permutation(make_sampler(n=8, seed=42, epoch=0))[3:]


def test_resolve_state_ignores_a_position_that_is_one_ahead():
    """accelerate's dataloader looks one batch ahead, so files written from the sampler counter are one too far."""

    sampler = make_sampler(n=8, seed=42)
    state = {"epoch": 0, "position": 4, "seed": 42, "num_samples": 8}  # step 3 completed
    assert resolve_state(state, sampler, current_step=3, current_epoch=1) == (0, 3)


def test_resolve_state_handles_the_real_state():
    """epoch 0 in the file, current_epoch 1 and step 1600 in train_state.json."""

    sampler = make_sampler(n=17521, seed=19260817)
    state = {"epoch": 0, "position": 1601, "seed": 19260817, "num_samples": 17521}
    assert resolve_state(state, sampler, current_step=1600, current_epoch=1) == (0, 1600)


def test_resolve_state_moves_an_epoch_boundary_state_to_the_next_epoch():
    sampler = make_sampler(n=8, seed=42)
    state = {"epoch": 0, "position": 8, "seed": 42, "num_samples": 8}
    assert resolve_state(state, sampler, current_step=8, current_epoch=1) == (1, 0)
    assert list(sampler) == permutation(make_sampler(n=8, seed=42, epoch=1))


def test_resolve_state_skips_the_finished_epochs():
    sampler = make_sampler(n=8, seed=42)
    state = {"epoch": 1, "position": 2, "seed": 42, "num_samples": 8}
    assert resolve_state(state, sampler, current_step=10, current_epoch=2) == (1, 2)
    assert list(sampler) == permutation(make_sampler(n=8, seed=42, epoch=1))[2:]


def test_resolve_state_counts_batches_with_gradient_accumulation():
    sampler = make_sampler(n=11, seed=42)
    state = {"epoch": 0, "position": 0, "seed": 42, "num_samples": 11}
    # 2 batches per step, 6 steps per epoch (11 batches): 3 steps = 6 batches
    assert resolve_state(state, sampler, current_step=3, current_epoch=1, gradient_accumulation_steps=2) == (0, 6)
    # the last step of the epoch is short: 6 steps = 11 batches, i.e. the epoch is over
    assert resolve_state(state, sampler, current_step=6, current_epoch=1, gradient_accumulation_steps=2) == (1, 0)


def test_resolve_state_continues_when_the_dataset_changed(caplog):
    """A growing dataset (added/removed images, a different `max_train_images` subset) cannot reproduce the order, but the
    run must still continue at the saved step instead of restarting."""

    sampler = make_sampler(n=17507, seed=42)
    state = {"epoch": 0, "position": 1601, "seed": 42, "num_samples": 17521}
    with caplog.at_level("WARNING"):
        assert resolve_state(state, sampler, current_step=1600, current_epoch=1) == (0, 1600)
    assert "17521" in caplog.text and "17507" in caplog.text


def test_resolve_state_continues_when_the_seed_changed(caplog):
    sampler = make_sampler(n=6, seed=43)
    with caplog.at_level("WARNING"):
        assert resolve_state({"seed": 42, "num_samples": 6}, sampler, current_step=2, current_epoch=1) == (0, 2)
    assert "seed" in caplog.text


def test_resolve_state_rejects_an_empty_dataset():
    sampler = make_sampler(n=0, seed=42)
    with pytest.raises(ValueError, match="empty dataset"):
        resolve_state({"seed": 42, "num_samples": 0}, sampler, current_step=1, current_epoch=1)


# endregion


# region integration: a resumed run continues the same batch and RNG stream


class RecordingDataset(Dataset):
    """Yields its index plus one draw from each RNG, so a test can compare the streams of two runs."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        return {"index": index, "python": random.random(), "torch": torch.rand(1).item()}


def make_loader(dataset: Dataset, sampler: DeterministicRandomSampler) -> DataLoader:
    # the dataloader draws one worker seed per epoch from its generator: a private one keeps the global torch RNG (noise,
    # timesteps) independent of how many epochs have been started, which is what makes a resume exact
    return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, generator=torch.Generator().manual_seed(0))


def collect(sampler, loader, epoch, records, limit=None):
    """Run one epoch like the trainer does: start the epoch unless it is already the resumed one, then consume batches.

    A training step draws noise after the batch, which is recorded too so the test also covers the RNG the model sees.
    """
    if sampler.epoch != epoch:
        sampler.set_epoch_position(epoch)
    for i, batch in enumerate(loader):
        records.append((batch["index"].item(), batch["python"].item(), batch["torch"].item(), torch.rand(1).item()))
        if limit is not None and i + 1 >= limit:
            break


def test_resume_reproduces_the_batch_and_rng_stream():
    n, epochs = 7, 3

    torch.manual_seed(1234)
    random.seed(1234)
    sampler = make_sampler(n=n, seed=42)
    loader = make_loader(RecordingDataset(n), sampler)
    expected = []
    for epoch in range(epochs):
        collect(sampler, loader, epoch, expected)
    assert len(expected) == n * epochs

    # the same run, aborted after 4 batches of epoch 1
    torch.manual_seed(1234)
    random.seed(1234)
    sampler = make_sampler(n=n, seed=42)
    loader = make_loader(RecordingDataset(n), sampler)
    interrupted = []
    collect(sampler, loader, 0, interrupted)
    collect(sampler, loader, 1, interrupted, limit=4)
    assert len(interrupted) == n + 4

    # what accelerate writes at that moment: the step/epoch (train_state.json) and the dataloader state
    saved_state = {"epoch": 1, "position": 5, "seed": 42, "num_samples": n}  # position from a v1 file: one too far
    saved_torch_rng = torch.get_rng_state()
    saved_python_rng = random.getstate()

    restored = make_sampler(n=n, seed=42, epoch=0)
    epoch, position = resolve_state(saved_state, restored, current_step=n + 4, current_epoch=2)
    assert (epoch, position) == (1, 4)
    torch.set_rng_state(saved_torch_rng)
    random.setstate(saved_python_rng)

    loader = make_loader(RecordingDataset(n), restored)
    resumed = []
    for e in range(epoch, epochs):
        collect(restored, loader, e, resumed)

    assert interrupted + resumed == expected  # same batches, same RNG draws, no repeated batch


def test_resume_from_an_epoch_boundary_state_starts_the_next_epoch():
    n = 5
    torch.manual_seed(99)
    random.seed(99)
    sampler = make_sampler(n=n, seed=7)
    loader = make_loader(RecordingDataset(n), sampler)
    expected = []
    collect(sampler, loader, 0, expected)
    saved_torch_rng = torch.get_rng_state()
    saved_python_rng = random.getstate()
    collect(sampler, loader, 1, expected)

    boundary = {"epoch": 0, "position": n, "seed": 7, "num_samples": n}
    restored = make_sampler(n=n, seed=7)
    assert resolve_state(boundary, restored, current_step=n, current_epoch=1) == (1, 0)
    torch.set_rng_state(saved_torch_rng)
    random.setstate(saved_python_rng)

    resumed = []
    collect(restored, make_loader(RecordingDataset(n), restored), 1, resumed)
    assert resumed == expected[n:]


# endregion
