"""Tests for global deterministic image selection (`max_train_images`)."""

import argparse
from pathlib import Path

from PIL import Image

from library.config_util import BlueprintGenerator, ConfigSanitizer, generate_dataset_group_by_blueprint


def make_images(root: Path, count: int = 12):
    for i in range(count):
        sub = root / ("a" if i < count // 2 else "b")
        sub.mkdir(exist_ok=True)
        image_path = sub / f"img{i:02d}.png"
        Image.new("RGB", (64 + i * 8, 80 + i * 8)).save(image_path)
        image_path.with_suffix(".txt").write_text(f"tag{i}", encoding="utf-8")


def make_config(root: Path, max_train_images: int = 5, seed: int = 123, validation_split: float = 0.0):
    config = {
        "general": {
            "max_train_images": max_train_images,
            "max_train_images_seed": seed,
            "recursive": True,
            "caption_extension": ".txt",
            "enable_bucket": True,
            "bucket_no_upscale": True,
            "bucket_reso_steps": 16,
        },
        "datasets": [
            {
                "resolution": 64,
                "validation_split": validation_split,
                "validation_seed": 0,
                "subsets": [{"image_dir": str(root)}],
            },
            {
                "resolution": 96,
                "validation_split": validation_split,
                "validation_seed": 0,
                "subsets": [{"image_dir": str(root)}],
            },
        ],
    }
    return config


def build(config):
    blueprint = BlueprintGenerator(ConfigSanitizer(True, False, False, True)).generate(config, argparse.Namespace())
    train_group, val_group = generate_dataset_group_by_blueprint(blueprint.dataset_group)
    return blueprint, train_group, val_group


def image_paths(dataset_group):
    return {
        str(Path(info.absolute_path).resolve())
        for dataset in dataset_group.datasets
        for info in dataset.image_data.values()
    }


def test_max_train_images_selects_same_images_across_all_datasets(tmp_path: Path):
    make_images(tmp_path)
    config = make_config(tmp_path)

    blueprint, train_group, _ = build(config)
    assert blueprint.dataset_group.params.max_train_images == 5
    assert blueprint.dataset_group.params.max_train_images_seed == 123

    paths_per_dataset = [
        {str(Path(info.absolute_path).resolve()) for info in dataset.image_data.values()}
        for dataset in train_group.datasets
    ]
    assert all(len(paths) == 5 for paths in paths_per_dataset)
    assert paths_per_dataset[0] == paths_per_dataset[1]

    for dataset in train_group.datasets:
        assert dataset.num_train_images == 5
        assert dataset.subsets[0].img_count == 5

    # the same seed must select the same images on a fresh run
    _, train_group2, _ = build(config)
    assert image_paths(train_group2) == image_paths(train_group)


def test_max_train_images_applies_to_train_and_validation_sets(tmp_path: Path):
    make_images(tmp_path)
    config = make_config(tmp_path, validation_split=0.5)

    _, train_group, val_group = build(config)
    assert val_group is not None

    all_paths = image_paths(train_group) | image_paths(val_group)
    assert len(all_paths) == 5
    assert image_paths(train_group).isdisjoint(image_paths(val_group))


def test_max_train_images_larger_than_pool_keeps_all_images(tmp_path: Path):
    make_images(tmp_path, count=4)
    config = make_config(tmp_path, max_train_images=100)

    _, train_group, _ = build(config)
    assert len(image_paths(train_group)) == 4
    assert all(dataset.num_train_images == 4 for dataset in train_group.datasets)
