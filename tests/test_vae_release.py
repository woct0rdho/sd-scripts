"""The VAE must be released once the latents are cached, and loaded only for the call that draws samples.

Keeping it resident costs its full size for the whole run, and loading it next to a training step would put the VAE and
the step's activations in the peak together. `to("meta")` releases the tensors, where `to("cpu")` only moves them.
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


class TinyVae(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(4, dim, 3, padding=1)
        self.norm = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.conv(x)


class VaeTrainer(train_network.NetworkTrainer):
    """A trainer with a working reloader, like the Anima and MOSS-Image ones."""

    def __init__(self, can_reload: bool = True):
        super().__init__()
        self.can_reload = can_reload
        self.loads = 0
        self.sampled_with = []

    def supports_vae_reload(self, args) -> bool:
        return self.can_reload

    def load_vae(self, args, dtype=None):
        self.loads += 1
        vae = TinyVae()
        return vae.to(dtype) if dtype is not None else vae

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        self.sampled_with.append(tensor_devices(vae))  # what the sampler sees, snapshotted during the call
        vae(torch.zeros(1, 4, 8, 8))  # the sampler runs the VAE, so it must be materialized


def cached_args(**overrides):
    """A config that caches the latents and draws samples on every step (step 10 with `--sample_every_n_steps 1`)."""
    args = dict(
        cache_latents=True,
        sample_at_first=False,
        sample_every_n_steps=1,
        sample_every_n_epochs=None,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def tensor_devices(module: nn.Module) -> set[str]:
    return {t.device.type for t in list(module.parameters()) + list(module.buffers())}


def test_vae_is_released_when_the_latents_are_cached():
    trainer = VaeTrainer()
    vae = TinyVae()
    trainer.release_vae_if_needed(cached_args(), FakeAccelerator(), vae)
    assert trainer._vae_released
    assert tensor_devices(vae) == {"meta"}  # the tensors are gone, not parked on the CPU


def test_vae_is_kept_without_latent_caching():
    trainer = VaeTrainer()
    vae = TinyVae()
    trainer.release_vae_if_needed(cached_args(cache_latents=False), FakeAccelerator(), vae)
    assert not trainer._vae_released
    assert tensor_devices(vae) == {"cpu"}


def test_vae_is_kept_when_the_family_cannot_reload_it():
    trainer = VaeTrainer(can_reload=False)
    vae = TinyVae()
    trainer.release_vae_if_needed(cached_args(), FakeAccelerator(), vae)
    assert not trainer._vae_released
    assert tensor_devices(vae) == {"cpu"}


def test_sampling_reloads_the_vae_and_releases_it_again():
    """The VAE is only resident inside the sampling call: one load per sample step, released after each."""
    trainer = VaeTrainer()
    vae = TinyVae()
    args = cached_args()
    trainer.release_vae_if_needed(args, FakeAccelerator(), vae)

    for _ in range(2):
        trainer.sample_images_with_vae(FakeAccelerator(), args, None, 10, torch.device("cpu"), vae, None, None, None)

    assert trainer.loads == 2  # once per sampling call, not kept
    assert trainer._vae_released  # and released again after the last one
    # every sampler invocation got a materialized VAE, and the cached `vae` object stayed released
    assert trainer.sampled_with == [{"cpu"}, {"cpu"}]
    assert tensor_devices(vae) == {"meta"}


def test_sampling_uses_the_resident_vae_without_caching():
    trainer = VaeTrainer()
    vae = TinyVae()
    args = cached_args(cache_latents=False)
    trainer.sample_images_with_vae(FakeAccelerator(), args, None, 10, torch.device("cpu"), vae, None, None, None)
    assert trainer.loads == 0
    assert trainer.sampled_with == [{"cpu"}]


def test_a_step_that_draws_nothing_leaves_the_vae_released():
    """The trainer calls the sampler on every step; only the scheduled ones may load the VAE."""
    trainer = VaeTrainer()
    vae = TinyVae()
    args = cached_args(sample_every_n_steps=500)
    trainer.release_vae_if_needed(args, FakeAccelerator(), vae)

    for step in (1, 2, 499):
        trainer.sample_images_with_vae(FakeAccelerator(), args, None, step, torch.device("cpu"), vae, None, None, None)
    assert trainer.loads == 0 and trainer.sampled_with == []
    assert trainer._vae_released

    trainer.sample_images_with_vae(FakeAccelerator(), args, None, 500, torch.device("cpu"), vae, None, None, None)
    assert trainer.loads == 1 and len(trainer.sampled_with) == 1
    assert trainer._vae_released  # and released again right after


# region sampling schedule


@pytest.mark.parametrize(
    "overrides,epoch,steps,expected",
    [
        ({"sample_at_first": True}, None, 0, True),
        ({"sample_at_first": False}, None, 0, False),
        ({"sample_every_n_steps": None, "sample_every_n_epochs": None}, None, 100, False),
        ({"sample_every_n_steps": 500}, None, 500, True),
        ({"sample_every_n_steps": 500}, None, 501, False),
        ({"sample_every_n_steps": 500}, 2, 500, False),  # the epoch-end call never draws by step
        ({"sample_every_n_steps": None, "sample_every_n_epochs": 2}, 2, 100, True),
        ({"sample_every_n_steps": None, "sample_every_n_epochs": 2}, 3, 100, False),
        ({"sample_every_n_steps": 1, "sample_every_n_epochs": 2}, 3, 100, False),
    ],
)
def test_should_sample_images(overrides, epoch, steps, expected):
    from library import sampling

    args = cached_args(**overrides)
    assert sampling.should_sample_images(args, epoch, steps) is expected


def test_a_failing_sample_still_releases_the_vae():
    """The release is in a `finally`: a sampler error must not leave the VAE resident for the next step."""

    class Broken(VaeTrainer):
        def sample_images(self, *args, **kwargs):
            raise RuntimeError("sampler failed")

    trainer = Broken()
    args = cached_args()
    with pytest.raises(RuntimeError, match="sampler failed"):
        trainer.sample_images_with_vae(FakeAccelerator(), args, None, 10, torch.device("cpu"), TinyVae(), None, None, None)
    assert trainer._vae_released


def test_the_released_dtype_is_restored():
    """A bf16 VAE must come back as bf16, or the sampler's casts would change what it draws."""
    trainer = VaeTrainer()
    vae = TinyVae().to(torch.bfloat16)
    trainer.release_vae_if_needed(cached_args(), FakeAccelerator(), vae)
    reloaded = trainer.prepare_vae_for_sampling(cached_args(), FakeAccelerator(), vae)
    assert {t.dtype for t in reloaded.parameters()} == {torch.bfloat16}
