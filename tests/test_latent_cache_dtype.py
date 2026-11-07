"""Tests for the latent cache storage dtype (`--latent_cache_dtype`) and tools/convert_latent_cache.py.

The cache is written in fp32, fp16 or bf16, independently of the VAE's dtype. npz has no bfloat16, so bf16 is stored as
the raw uint16 bit pattern; loading must come back as a bfloat16 tensor *view*, without copying or a float32 round trip.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch

from library import caching

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "convert_latent_cache.py")


def load_tool():
    spec = importlib.util.spec_from_file_location("convert_latent_cache", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def restore_storage_dtype():
    previous = caching.get_storage_dtype()
    LatentsCachingStrategy._warned_storage_dtype_mismatch = False
    yield
    caching.set_storage_dtype(previous)
    LatentsCachingStrategy._warned_storage_dtype_mismatch = False


# region helpers


@pytest.mark.parametrize("name", caching.LATENT_STORAGE_DTYPES)
def test_storage_round_trip_is_exact_and_zero_copy(name):
    tensor = torch.randn(2, 32, 6, 8)
    array = caching.latent_to_storage(tensor, name)
    back = caching.latent_from_storage(array)
    expected = tensor if name == "fp32" else tensor.to(torch.bfloat16 if name == "bf16" else torch.float16)
    assert back.dtype == expected.dtype
    assert torch.equal(back, expected)
    assert back.data_ptr() == array.ctypes.data  # a view, not a copy


def test_storage_dtype_of_the_np_arrays():
    assert caching.storage_np_dtype("fp32") == np.dtype(np.float32)
    assert caching.storage_np_dtype("fp16") == np.dtype(np.float16)
    assert caching.storage_np_dtype("bf16") == np.dtype(np.uint16)  # npz has no bfloat16


def test_set_storage_dtype_validates():
    caching.set_storage_dtype("bf16")
    assert caching.get_storage_dtype() == "bf16"
    with pytest.raises(ValueError, match="unknown latent cache dtype"):
        caching.set_storage_dtype("fp8")


def test_non_contiguous_input_is_made_contiguous():
    tensor = torch.randn(2, 6, 32, 8).transpose(1, 2)
    array = caching.latent_to_storage(tensor, "bf16")
    assert torch.equal(caching.latent_from_storage(array), tensor.to(torch.bfloat16))


# endregion


# region strategy


from library.strategy_base import LatentsCachingStrategy  # noqa: E402


class TestStrategy(LatentsCachingStrategy):
    """A concrete strategy with stride 8 and multi-resolution keys, like the model-specific ones."""

    @property
    def cache_suffix(self) -> str:
        return "_test.npz"

    def get_latents_npz_path(self, absolute_path, image_size):
        return absolute_path + self.cache_suffix

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, True)

    def load_latents_from_disk(self, npz_path, bucket_reso):
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)


def _strategy():
    return TestStrategy(True, 1, False)


def save_npz(tmp_path, name, latents, flipped=None, dtype=None, marker=True, reso="6x8"):
    kwargs = {f"latents_{reso}": caching.latent_to_storage(latents, dtype), "original_size_" + reso: np.array([48, 64])}
    kwargs["crop_ltrb_" + reso] = np.array([0, 0, 0, 0])
    if flipped is not None:
        kwargs["latents_flipped_" + reso] = caching.latent_to_storage(flipped, dtype)
    if marker:
        kwargs[caching.STORAGE_DTYPE_KEY] = np.array(dtype or "fp32")
    path = str(tmp_path / name)
    np.savez(path, **kwargs)
    return path


@pytest.mark.parametrize("name", caching.LATENT_STORAGE_DTYPES)
def test_save_and_load_use_the_configured_dtype(tmp_path, name):
    caching.set_storage_dtype(name)
    strategy = _strategy()
    latents = torch.randn(32, 6, 8)
    flipped = torch.randn(32, 6, 8)
    path = str(tmp_path / "a.npz")

    strategy.save_latents_to_disk(path, latents, [48, 64], [0, 0, 0, 0], flipped, None, "_6x8")

    npz = np.load(path)
    assert npz[caching.STORAGE_DTYPE_KEY] == np.array(name)
    assert npz["latents_6x8"].dtype == caching.storage_np_dtype(name)
    assert npz["latents_flipped_6x8"].dtype == caching.storage_np_dtype(name)

    loaded, original_size, crop, loaded_flipped, alpha = strategy.load_latents_from_disk(path, (64, 48))
    expected = latents if name == "fp32" else latents.to(caching.latent_from_storage(npz["latents_6x8"]).dtype)
    assert torch.equal(loaded, expected)
    assert torch.equal(loaded_flipped, flipped.to(loaded.dtype))
    assert original_size == [48, 64] and crop == [0, 0, 0, 0] and alpha is None

    bucket_reso = (64, 48)
    assert strategy.is_disk_cached_latents_expected(bucket_reso, path, flip_aug=True, alpha_mask=False)


def test_the_marker_rejects_a_cache_written_in_another_dtype(tmp_path):
    strategy = _strategy()
    latents = torch.randn(32, 6, 8)
    path = save_npz(tmp_path, "fp32.npz", latents, dtype="fp32")

    caching.set_storage_dtype("fp32")
    assert strategy.is_disk_cached_latents_expected((64, 48), path, flip_aug=False, alpha_mask=False)

    caching.set_storage_dtype("bf16")
    assert not strategy.is_disk_cached_latents_expected((64, 48), path, flip_aug=False, alpha_mask=False)


def test_a_cache_without_the_marker_is_fp32(tmp_path):
    strategy = _strategy()
    path = save_npz(tmp_path, "old.npz", torch.randn(32, 6, 8), dtype="fp32", marker=False)

    caching.set_storage_dtype("fp32")
    assert strategy.is_disk_cached_latents_expected((64, 48), path, flip_aug=False, alpha_mask=False)
    caching.set_storage_dtype("bf16")
    assert not strategy.is_disk_cached_latents_expected((64, 48), path, flip_aug=False, alpha_mask=False)


# endregion


# region conversion tool


def make_cache(tmp_path, name, dtype="fp32", subdir="", resos=("6x8",)):
    directory = tmp_path / subdir if subdir else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    for reso in resos:
        h, w = (int(v) for v in reso.split("x"))
        latents = torch.randn(32, h, w)
        kwargs["latents_" + reso] = caching.latent_to_storage(latents, dtype)
        kwargs["latents_flipped_" + reso] = caching.latent_to_storage(torch.randn(32, h, w), dtype)
        kwargs["original_size_" + reso] = np.array([w * 8, h * 8])
        kwargs["crop_ltrb_" + reso] = np.array([0, 0, 0, 0])
    kwargs[caching.STORAGE_DTYPE_KEY] = np.array(dtype)
    path = directory / name
    np.savez(path, **kwargs)
    return path


def run_tool(*args):
    tool = load_tool()
    import sys

    argv = sys.argv
    sys.argv = ["convert_latent_cache.py", *args]
    try:
        return tool.main()
    finally:
        sys.argv = argv


def test_tool_converts_in_place_and_is_idempotent(tmp_path):
    path = make_cache(tmp_path, "a_anima.npz", "fp32")
    before = np.load(path)
    fp32 = {k: before[k] for k in before.files}

    assert run_tool("--dir", str(tmp_path), "--to", "bf16") == 0
    npz = np.load(path)
    assert str(npz[caching.STORAGE_DTYPE_KEY]) == "bf16"
    assert npz["latents_6x8"].dtype == np.uint16
    # the values are exactly the fp32 ones rounded to bf16, and every other key survived
    assert torch.equal(
        caching.latent_from_storage(npz["latents_6x8"]), torch.from_numpy(fp32["latents_6x8"]).to(torch.bfloat16)
    )
    assert np.array_equal(npz["original_size_6x8"], fp32["original_size_6x8"])
    assert np.array_equal(npz["crop_ltrb_6x8"], fp32["crop_ltrb_6x8"])

    # a second run finds nothing to do
    assert run_tool("--dir", str(tmp_path), "--to", "bf16") == 0
    assert str(np.load(path)[caching.STORAGE_DTYPE_KEY]) == "bf16"

    # and fp16 -> fp32 restores full precision of the fp16 values
    assert run_tool("--dir", str(tmp_path), "--to", "fp16") == 0
    assert run_tool("--dir", str(tmp_path), "--to", "fp32") == 0
    npz = np.load(path)
    assert npz["latents_6x8"].dtype == np.float32
    expected = torch.from_numpy(fp32["latents_6x8"]).to(torch.bfloat16).to(torch.float16).float()
    assert torch.equal(torch.from_numpy(npz["latents_6x8"]), expected)


def test_tool_is_not_recursive_unless_asked(tmp_path):
    make_cache(tmp_path, "top.npz", "fp32")
    nested = make_cache(tmp_path, "nested.npz", "fp32", subdir="sub/deeper")
    before = os.path.getsize(nested)

    assert run_tool("--dir", str(tmp_path), "--to", "bf16") == 0
    assert str(np.load(nested)[caching.STORAGE_DTYPE_KEY]) == "fp32"  # untouched
    assert os.path.getsize(nested) == before

    assert run_tool("--dir", str(tmp_path), "--recursive", "--to", "bf16") == 0
    assert str(np.load(nested)[caching.STORAGE_DTYPE_KEY]) == "bf16"
    assert str(np.load(tmp_path / "top.npz")[caching.STORAGE_DTYPE_KEY]) == "bf16"


def test_tool_dry_run_writes_nothing(tmp_path):
    path = make_cache(tmp_path, "a_moss.npz", "fp32")
    before = os.path.getsize(path)
    assert run_tool("--dir", str(tmp_path), "--to", "bf16", "--dry-run") == 0
    assert os.path.getsize(path) == before
    assert str(np.load(path)[caching.STORAGE_DTYPE_KEY]) == "fp32"


def test_tool_skips_files_without_latents(tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, data=np.zeros(3))
    size = os.path.getsize(path)
    assert run_tool("--dir", str(tmp_path), "--to", "bf16") == 0
    assert os.path.getsize(path) == size  # not rewritten


def test_tool_reports_no_files(tmp_path):
    assert run_tool("--dir", str(tmp_path), "--to", "bf16") == 1


# endregion
