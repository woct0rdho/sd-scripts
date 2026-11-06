"""Tests for skipping corrupted images whose size cannot be determined."""

import argparse
from pathlib import Path

from PIL import Image
import torch

from library.config_util import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint
from library.dataset import BaseDataset
from library.strategy_base import LatentsCachingStrategy


def make_images(root: Path, valid_count: int = 10, add_corrupt: bool = True):
    for i in range(valid_count):
        image_path = root / f"img{i:02d}.png"
        Image.new("RGB", (64 + i * 2, 80 + i * 2)).save(image_path)
        image_path.with_suffix(".txt").write_text(f"tag{i}", encoding="utf-8")

    if add_corrupt:
        (root / "broken.png").write_bytes(b"this is not a valid png file")
        (root / "broken.txt").write_text("broken", encoding="utf-8")


def make_config(root: Path, max_train_images=None):
    general = {
        "caption_extension": ".txt",
        "enable_bucket": True,
        "bucket_no_upscale": True,
        "bucket_reso_steps": 16,
    }
    if max_train_images is not None:
        general["max_train_images"] = max_train_images
        general["max_train_images_seed"] = 123
    return {"general": general, "datasets": [{"resolution": 64, "subsets": [{"image_dir": str(root)}]}]}


def build(config):
    blueprint = BlueprintGenerator(ConfigSanitizer(True, False, False, True)).generate(config, argparse.Namespace())
    return generate_dataset_group_by_blueprint(blueprint.dataset_group)


def test_get_image_size_returns_zero_for_corrupt_image(tmp_path: Path):
    make_images(tmp_path, valid_count=0)
    dataset = BaseDataset((64, 64), 1.0, False, False)
    assert dataset.get_image_size(str(tmp_path / "broken.png")) == (0, 0)


def test_corrupt_image_is_skipped_before_bucketing(tmp_path: Path):
    make_images(tmp_path)
    train_group, _ = build(make_config(tmp_path))
    dataset = train_group.datasets[0]

    image_names = {Path(info.absolute_path).name for info in dataset.image_data.values()}
    assert "broken.png" not in image_names
    assert len(image_names) == 10
    assert dataset.num_train_images == 10
    assert dataset.subsets[0].img_count == 10
    assert dataset.bucket_info is not None


def test_corrupt_images_are_excluded_before_max_train_images_selection(tmp_path: Path):
    make_images(tmp_path)
    train_group, _ = build(make_config(tmp_path, max_train_images=10))
    dataset = train_group.datasets[0]

    image_names = {Path(info.absolute_path).name for info in dataset.image_data.values()}
    assert "broken.png" not in image_names
    assert len(image_names) == 10  # all valid images are kept, despite the corrupt one

    train_group, _ = build(make_config(tmp_path, max_train_images=5))
    dataset = train_group.datasets[0]
    image_names = {Path(info.absolute_path).name for info in dataset.image_data.values()}
    assert "broken.png" not in image_names
    assert len(image_names) == 5
    assert dataset.subsets[0].img_count == 5


class FakeCachingStrategy:
    cache_to_disk = True
    batch_size = 2
    cache_suffix = ".npz"

    def __init__(self):
        self.cached_infos = []

    def get_latents_npz_path(self, absolute_path, image_size):
        return absolute_path + ".npz"

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        return False

    def cache_batch_latents(self, model, batch, flip_aug, alpha_mask, random_crop):
        self.cached_infos.extend(batch)


class FakeAccelerator:
    num_processes = 1
    process_index = 0
    device = torch.device("cpu")

    def wait_for_everyone(self):
        pass


def test_truncated_image_is_skipped_during_latent_caching(tmp_path: Path, monkeypatch):
    make_images(tmp_path, valid_count=4, add_corrupt=False)

    # this image has a readable size (header), but decoding the pixel data fails
    truncated_path = tmp_path / "truncated.png"
    Image.new("RGB", (128, 96), color=(255, 0, 0)).save(truncated_path)
    data = truncated_path.read_bytes()
    truncated_path.write_bytes(data[:-100])
    truncated_path.with_suffix(".txt").write_text("truncated", encoding="utf-8")
    assert BaseDataset((64, 64), 1.0, False, False).get_image_size(str(truncated_path)) == (128, 96)

    strategy = FakeCachingStrategy()
    monkeypatch.setattr(LatentsCachingStrategy, "get_strategy", classmethod(lambda cls: strategy))

    train_group, _ = build(make_config(tmp_path))
    dataset = train_group.datasets[0]
    assert truncated_path.name in {Path(info.absolute_path).name for info in dataset.image_data.values()}

    dataset.new_cache_latents(None, FakeAccelerator())

    image_names = {Path(info.absolute_path).name for info in dataset.image_data.values()}
    assert truncated_path.name not in image_names
    assert len(image_names) == 4
    assert dataset.num_train_images == 4
    assert dataset.subsets[0].img_count == 4
    assert {Path(info.absolute_path).name for info in strategy.cached_infos} == image_names
    assert len(dataset) > 0


def test_concat_dataset_lengths_are_refreshed_after_cache_removal(tmp_path: Path, monkeypatch):
    make_images(tmp_path, valid_count=4, add_corrupt=False)

    truncated_path = tmp_path / "truncated.png"
    Image.new("RGB", (128, 96), color=(255, 0, 0)).save(truncated_path)
    data = truncated_path.read_bytes()
    truncated_path.write_bytes(data[:-100])
    truncated_path.with_suffix(".txt").write_text("truncated", encoding="utf-8")

    strategy = FakeCachingStrategy()
    monkeypatch.setattr(LatentsCachingStrategy, "get_strategy", classmethod(lambda cls: strategy))

    config = make_config(tmp_path)
    config["datasets"].append({"resolution": 96, "subsets": [{"image_dir": str(tmp_path)}]})
    train_group, _ = build(config)
    assert train_group.cumulative_sizes == [5, 10]

    train_group.new_cache_latents(None, FakeAccelerator())

    assert [len(dataset) for dataset in train_group.datasets] == [4, 4]
    assert train_group.cumulative_sizes == [4, 8]
    assert len(train_group) == 8
