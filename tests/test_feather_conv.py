"""Tests for library/feather_conv.py (the VAE 3x3 conv wiring).

The interesting part is not that the kernel runs, it is that everything else stays the same: ineligible
modules keep torch's forward (bit-identically), the module tree, `state_dict` keys, dtypes, parameters and
`load_state_dict` are untouched, the patch is idempotent, and the prepack follows the weight version.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from library import feather_conv
from library.qwen_image_autoencoder_kl import ChunkedConv2d

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def patch_backend(monkeypatch):
    """Force the feather backend on (the structure tests must not depend on the machine's extension)."""

    def set_backend(backend):
        monkeypatch.setattr(feather_conv, "conv_backend", lambda: backend)

    return set_backend


def conv3x3(c_in, c_out, bias=True):
    return nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=bias)


class VaeLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = conv3x3(16, 32)
        self.block = nn.Sequential(conv3x3(32, 32), nn.SiLU(), conv3x3(32, 16))
        self.conv_out = nn.Conv2d(16, 3, 3, stride=1, padding=1)  # C_out below the tile minimum
        self.nin = nn.Conv2d(16, 16, 1)  # 1x1: not in scope
        self.down = nn.Conv2d(3, 3, 3, stride=2, padding=0)  # stride 2: not in scope

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block(x)
        x = self.nin(x)
        x = self.conv_out(x)
        return self.down(x)


# region geometry / structure


def test_candidate_geometry():
    assert feather_conv._is_candidate(conv3x3(8, 16))
    assert feather_conv._is_candidate(ChunkedConv2d(8, 16, 3, padding=1))
    assert not feather_conv._is_candidate(nn.Conv2d(8, 16, 1))
    assert not feather_conv._is_candidate(nn.Conv2d(8, 16, 3, stride=2, padding=1))
    assert not feather_conv._is_candidate(nn.Conv2d(8, 16, 3, padding=0))
    assert not feather_conv._is_candidate(nn.Conv2d(8, 16, 3, padding=1, dilation=2))
    assert not feather_conv._is_candidate(nn.Conv2d(8, 16, 3, padding=1, groups=2))
    assert not feather_conv._is_candidate(nn.Conv2d(8, 16, 3, padding=1, padding_mode="reflect"))
    assert not feather_conv._is_candidate(nn.Linear(8, 16))

    class CustomConv(nn.Conv2d):
        def forward(self, x):  # a subclass with its own forward is left alone
            return super().forward(x) * 2

    assert not feather_conv._is_candidate(CustomConv(8, 16, 3, padding=1))


def test_patch_is_a_noop_on_the_torch_backend(patch_backend):
    patch_backend("torch")
    model = VaeLike()
    before = [type(m) for m in model.modules()]
    assert feather_conv.patch_conv_modules(model) == {"candidates": 0, "patched": 0}
    assert [type(m) for m in model.modules()] == before
    assert not feather_conv.is_patched(model)


def test_patch_relabels_only_the_eligible_convs(patch_backend):
    patch_backend("feather-ops")
    model = VaeLike()
    conv_in, conv1, conv2 = model.conv_in, model.block[0], model.block[2]
    state_keys = list(model.state_dict().keys())
    params = {name: param for name, param in model.named_parameters()}

    stats = feather_conv.patch_conv_modules(model)

    assert stats == {"candidates": 4, "patched": 4}  # conv_in, block[0], block[2], conv_out
    for module in (conv_in, conv1, conv2, model.conv_out):
        assert isinstance(module, feather_conv._FeatherConv2dMixin)
        assert isinstance(module, nn.Conv2d)  # the model tree still holds plain convs
    assert not isinstance(model.nin, feather_conv._FeatherConv2dMixin)
    assert not isinstance(model.down, feather_conv._FeatherConv2dMixin)
    assert feather_conv.is_patched(model)

    # nothing but the class changed
    assert list(model.state_dict().keys()) == state_keys
    assert {name: param for name, param in model.named_parameters()} == params
    assert model.conv_in.weight.dtype == torch.float32

    # idempotent, and the relabelled class is cached
    assert feather_conv.patch_conv_modules(model) == {"candidates": 4, "patched": 0}
    assert type(model.conv_in) is feather_conv._relabelled(nn.Conv2d)


def test_patched_model_keeps_load_state_dict_and_forward_shape(patch_backend):
    patch_backend("feather-ops")
    torch.manual_seed(0)
    patched = VaeLike()
    feather_conv.patch_conv_modules(patched)
    plain = VaeLike()
    plain.load_state_dict(patched.state_dict())  # keys are unchanged
    x = torch.randn(2, 16, 8, 8)
    assert torch.equal(patched(x), plain(x))  # fp32 on cpu is a fallback everywhere: bit-identical
    assert plain.state_dict().keys() == patched.state_dict().keys()


def test_chunked_conv_padding_is_understood(patch_backend):
    patch_backend("feather-ops")
    module = ChunkedConv2d(8, 16, 3, padding=1)
    assert feather_conv._is_candidate(module)
    assert tuple(feather_conv._sub_padding(module)) == (1, 1)  # not the zeroed `padding`
    assert not feather_conv._is_candidate(ChunkedConv2d(8, 16, 3, stride=2))


# endregion


# region kernel / delegation routing


@cuda_only
def test_bf16_3x3_conv_uses_the_kernel(monkeypatch):
    import feather_ops.conv2d as feather_ops_conv2d

    calls = []
    real = feather_ops_conv2d.conv3x3_prepacked
    monkeypatch.setattr(feather_ops_conv2d, "conv3x3_prepacked", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    module = conv3x3(64, 128).cuda().to(torch.bfloat16)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(1, 64, 32, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = module(x)

    assert len(calls) == 1
    reference = F.conv2d(x, module.weight, module.bias, 1, 1).float()
    relative = (out.float() - reference).norm() / reference.norm()
    assert relative < 4e-3  # bf16 arithmetic, not bit-exact (the reduction order differs)


@cuda_only
def test_fp16_is_delegated_to_miopen(monkeypatch):
    import feather_ops.conv2d as feather_ops_conv2d

    calls = []
    real = feather_ops_conv2d.conv3x3_prepacked
    monkeypatch.setattr(feather_ops_conv2d, "conv3x3_prepacked", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    module = conv3x3(64, 128).cuda().to(torch.float16)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(1, 64, 32, 32, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        out = module(x)

    assert calls == []  # MIOpen's Winograd wins at fp16 on this part
    reference = F.conv2d(x, module.weight, module.bias, 1, 1).float()
    relative = (out.float() - reference).norm() / reference.norm()
    assert relative < 6e-4


@cuda_only
def test_untiled_shapes_are_delegated_to_miopen(monkeypatch):
    import feather_ops.conv2d as feather_ops_conv2d

    calls = []
    real = feather_ops_conv2d.conv3x3_prepacked
    monkeypatch.setattr(feather_ops_conv2d, "conv3x3_prepacked", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    assert feather_ops_conv2d.select_backend(torch.bfloat16, 96, 3, 64, 64) == "miopen"  # no tile divides C_out=3
    module = nn.Conv2d(96, 3, 3, padding=1).cuda().to(torch.bfloat16)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(1, 96, 64, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = module(x)

    assert calls == []
    assert out.shape == (1, 3, 64, 64)


@cuda_only
def test_small_channels_use_miopen_and_the_output_is_right():
    # the first encoder conv: C_in % 8 != 0, so the kernel cannot run
    module = nn.Conv2d(3, 96, 3, padding=1).cuda().to(torch.bfloat16)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = module(x)
    reference = F.conv2d(x, module.weight, module.bias, 1, 1).float()
    assert (out.float() - reference).norm() / reference.norm() < 4e-3


# endregion


# region fallbacks stay torch's (and stay correct)


@cuda_only
@pytest.mark.parametrize("case", ["fp32", "grad_enabled", "non_contiguous"])
def test_fallbacks_are_bit_identical_to_torch(patch_backend, case):
    patch_backend("feather-ops")
    torch.manual_seed(0)
    dtype = torch.bfloat16 if case != "fp32" else torch.float32
    module = conv3x3(16, 32).cuda().to(dtype)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(2, 16, 16, 16, device="cuda", dtype=dtype)
    if case == "non_contiguous":
        x = x.transpose(2, 3)

    if case == "grad_enabled":
        out = module(x)  # grad mode on: the fast path must decline
    else:
        with torch.no_grad():
            out = module(x)
    reference = F.conv2d(x, module.weight, module.bias, 1, 1)
    assert torch.equal(out, reference)


@cuda_only
def test_a_bias_that_does_not_match_the_activation_declines(patch_backend):
    # torch itself rejects this input, so the fast path must simply not take it (the module's own forward decides)
    patch_backend("feather-ops")
    module = conv3x3(16, 32).cuda()
    module.bias.data = module.bias.data.float()
    module.weight.data = module.weight.data.to(torch.bfloat16)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(1, 16, 16, 16, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert feather_conv._try_feather_conv2d(module, x) is None


@cuda_only
def test_chunked_conv_routes_and_keeps_its_chunking(patch_backend):
    patch_backend("feather-ops")
    torch.manual_seed(0)
    module = ChunkedConv2d(16, 32, 3, padding=1).cuda().to(torch.bfloat16)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(1, 16, 96, 96, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        unchunked = module(x)
    reference = F.conv2d(x, module.weight, module.bias, 1, 1).float()
    assert (unchunked.float() - reference).norm() / reference.norm() < 4e-3

    # a chunk size that actually engages the chunked path: the module keeps its own forward
    module.spatial_chunk_size = 16
    assert feather_conv._try_feather_conv2d(module, x) is None
    with torch.no_grad():
        chunked = module(x)
    assert (chunked.float() - reference).norm() / reference.norm() < 4e-3


# endregion


# region prepack lifetime


@cuda_only
def test_prepack_is_cached_and_follows_the_weight_version(monkeypatch):
    import feather_ops.conv2d as feather_ops_conv2d

    packs = []
    real = feather_ops_conv2d.prepack_conv2d_weight_3x3
    monkeypatch.setattr(feather_ops_conv2d, "prepack_conv2d_weight_3x3", lambda *a, **k: (packs.append(1), real(*a, **k))[1])

    module = conv3x3(32, 64).cuda().to(torch.bfloat16)
    feather_conv.patch_conv_modules(module)
    x = torch.randn(1, 32, 32, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        module(x)
        module(x)
    assert len(packs) == 1  # one pack for the module

    with torch.no_grad():
        module.weight.mul_(1.0)  # in-place update: bumps the version, so the prepack is rebuilt
        module(x)
    assert len(packs) == 2

    module.load_state_dict({"weight": module.weight.clone(), "bias": module.bias.clone()})
    with torch.no_grad():
        module(x)
    assert len(packs) == 3

    # a cast drops the cached prepack instead of keeping a stale copy
    module.to(torch.float16)
    assert "_feather_conv_packed" not in module.__dict__


# endregion


# region backend selection


def test_backend_env(monkeypatch):
    from library import feather_conv as module

    monkeypatch.setattr(module, "_backend_logged", False, raising=False)
    monkeypatch.setenv(feather_conv.BACKEND_ENV, "torch")
    assert module.conv_backend() == "torch"

    monkeypatch.setattr(module, "_backend_logged", False, raising=False)
    monkeypatch.setenv(feather_conv.BACKEND_ENV, "bogus")
    with pytest.raises(ValueError, match="unknown"):
        module.conv_backend()

    monkeypatch.setattr(module, "_backend_logged", False, raising=False)
    monkeypatch.setenv(feather_conv.BACKEND_ENV, "auto")
    expected = "feather-ops" if module.feather_conv_available() else "torch"
    assert module.conv_backend() == expected


def test_feather_ops_backend_is_required_when_requested(monkeypatch):
    from library import feather_conv as module

    monkeypatch.setattr(module, "_backend_logged", False, raising=False)
    monkeypatch.setattr(module, "feather_conv_available", lambda: False)
    monkeypatch.setenv(feather_conv.BACKEND_ENV, "feather-ops")
    with pytest.raises(RuntimeError, match="not usable"):
        module.conv_backend()


@cuda_only
def test_patch_on_the_real_extension_with_bf16_vae():
    if not feather_conv.feather_conv_available():
        pytest.skip("feather_ops.conv2d is not available")
    torch.manual_seed(0)
    model = VaeLike().cuda().to(torch.bfloat16)
    stats = feather_conv.patch_conv_modules(model)
    assert stats["patched"] == 4
    x = torch.randn(1, 16, 32, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(x)
    reference = VaeLike().cuda().to(torch.bfloat16)
    reference.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected = reference(x)
    relative = (out.float() - expected.float()).norm() / expected.float().norm()
    assert relative < 1e-2


# endregion


# region --vae_dtype


def test_resolve_vae_dtype():
    from library import accelerator_setup

    assert accelerator_setup.resolve_vae_dtype(argparse_namespace("auto"), torch.bfloat16) is torch.bfloat16
    assert accelerator_setup.resolve_vae_dtype(argparse_namespace(None), torch.float32) is torch.float32
    assert accelerator_setup.resolve_vae_dtype(argparse_namespace("bf16"), torch.float32) is torch.bfloat16
    assert accelerator_setup.resolve_vae_dtype(argparse_namespace("fp16"), torch.bfloat16) is torch.float16
    assert accelerator_setup.resolve_vae_dtype(argparse_namespace("fp32"), torch.bfloat16) is torch.float32
    assert set(accelerator_setup.VAE_DTYPES) == {"fp32", "fp16", "bf16"}


def argparse_namespace(vae_dtype):
    import argparse

    return argparse.Namespace(vae_dtype=vae_dtype)


def test_vae_dtype_argument_is_registered():
    import argparse

    from library import args as args_util

    parser = argparse.ArgumentParser()
    args_util.add_training_arguments(parser, support_dreambooth=True)
    assert parser.parse_args([]).vae_dtype == "auto"
    assert parser.parse_args(["--vae_dtype", "fp16"]).vae_dtype == "fp16"
    with pytest.raises(SystemExit):
        parser.parse_args(["--vae_dtype", "int8"])


# endregion
