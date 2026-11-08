"""The text encoders must be *released* after their outputs are cached, not just moved to the CPU.

`Module.to("cpu")` only relocates the tensors: on a unified-memory machine host and device memory are the same RAM, so
the cached text encoder still occupies its GBs and cannot be reused for the model. `to("meta")` drops them.
"""

import argparse
import contextlib

import pytest
import torch
import torch.nn as nn

import train_network


class FakeAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model


class TinyTextEncoder(nn.Module):
    """Stands in for a text encoder: two 1024-wide bf16 tensors and one int8 (ConvRot) weight."""

    def __init__(self, dim: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(32, dim, dtype=torch.bfloat16)
        self.proj = nn.Linear(dim, dim, bias=False, dtype=torch.bfloat16)
        self.int8 = nn.Parameter(torch.zeros(dim, dim, dtype=torch.int8), requires_grad=False)

    def forward(self, ids):
        return self.proj(self.embed(ids).to(self.proj.weight.dtype))


class CachingTrainer(train_network.NetworkTrainer):
    """A trainer that caches the text encoder outputs and encodes nothing afterwards (the Anima/MOSS shape)."""

    def __init__(self, needs_encoders_later: bool = False, trains_text_encoder: bool = False):
        super().__init__()
        self.needs_encoders_later = needs_encoders_later
        self.trains_text_encoder = trains_text_encoder

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs and not self.needs_encoders_later:
            return None  # nothing encodes with them any more
        return text_encoders

    def is_train_text_encoder(self, args):
        return self.trains_text_encoder


def tensors_on(module: nn.Module, device_type: str) -> int:
    return sum(1 for t in list(module.parameters()) + list(module.buffers()) if t.device.type == device_type)


def all_tensors(module: nn.Module) -> int:
    return len(list(module.parameters())) + len(list(module.buffers()))


def test_cached_text_encoder_is_moved_to_meta():
    trainer = CachingTrainer()
    te = TinyTextEncoder()
    args = argparse.Namespace(cache_text_encoder_outputs=True)

    trainer.release_text_encoders_if_needed(args, FakeAccelerator(), [te])

    # every tensor is gone (meta), nothing was left on the CPU
    assert tensors_on(te, "cpu") == 0
    assert tensors_on(te, "meta") == all_tensors(te) > 0


def test_text_encoder_is_kept_when_the_trainer_still_needs_it():
    """The SDXL shape: `get_models_for_text_encoding` keeps returning the encoders, so they stay intact."""
    trainer = CachingTrainer(needs_encoders_later=True)
    te = TinyTextEncoder()
    trainer.release_text_encoders_if_needed(argparse.Namespace(cache_text_encoder_outputs=True), FakeAccelerator(), [te])
    assert tensors_on(te, "meta") == 0
    assert tensors_on(te, "cpu") == all_tensors(te)


def test_text_encoder_is_kept_without_caching():
    trainer = CachingTrainer()
    te = TinyTextEncoder()
    trainer.release_text_encoders_if_needed(argparse.Namespace(cache_text_encoder_outputs=False), FakeAccelerator(), [te])
    assert tensors_on(te, "meta") == 0


def test_text_encoder_is_kept_when_it_is_being_trained():
    """Caching plus a text encoder LoRA would need its gradients, so it must not be released."""
    trainer = CachingTrainer(trains_text_encoder=True)
    te = TinyTextEncoder()
    trainer.release_text_encoders_if_needed(argparse.Namespace(cache_text_encoder_outputs=True), FakeAccelerator(), [te])
    assert tensors_on(te, "meta") == 0


def test_release_is_idempotent_and_skips_none_entries():
    trainer = CachingTrainer()
    te = TinyTextEncoder()
    args = argparse.Namespace(cache_text_encoder_outputs=True)
    trainer.release_text_encoders_if_needed(args, FakeAccelerator(), [te, None])
    trainer.release_text_encoders_if_needed(args, FakeAccelerator(), [te])  # meta -> meta is a no-op
    assert tensors_on(te, "meta") == all_tensors(te)


@pytest.mark.parametrize("needs_later", [False, True])
def test_release_does_not_break_a_later_forward(needs_later):
    """Whatever is kept must still run; the released one is never called again by construction."""
    trainer = CachingTrainer(needs_encoders_later=needs_later)
    te = TinyTextEncoder()
    before = te(torch.zeros(2, dtype=torch.long)).detach().clone()
    trainer.release_text_encoders_if_needed(argparse.Namespace(cache_text_encoder_outputs=True), FakeAccelerator(), [te])
    if needs_later:
        assert torch.equal(te(torch.zeros(2, dtype=torch.long)).detach(), before)
    else:
        assert tensors_on(te, "cpu") == 0  # would raise on a forward, which is the point
