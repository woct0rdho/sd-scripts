"""Tests for recursive DreamBooth image scanning (`recursive = true`)."""

import argparse
from pathlib import Path

from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.dataset import glob_images


def test_glob_images_recursive(tmp_path: Path):
    (tmp_path / "a" / "deep").mkdir(parents=True)
    for path in [tmp_path / "root.png", tmp_path / "a" / "one.jpg", tmp_path / "a" / "deep" / "two.webp"]:
        path.write_bytes(b"dummy")
    (tmp_path / "a" / "not_an_image.txt").write_text("tag", encoding="utf-8")

    flat = {Path(p).relative_to(tmp_path).as_posix() for p in glob_images(str(tmp_path), "*")}
    recursive = {Path(p).relative_to(tmp_path).as_posix() for p in glob_images(str(tmp_path), "*", recursive=True)}

    assert flat == {"root.png"}
    assert recursive == {"root.png", "a/one.jpg", "a/deep/two.webp"}


def test_dataset_config_recursive_is_passed_to_subset(tmp_path: Path):
    config = {
        "general": {
            "recursive": True,
            "caption_extension": ".txt",
            "shuffle_caption": True,
            "caption_tag_dropout_rate": 0.5,
            "flip_aug": True,
            "enable_bucket": True,
        },
        "datasets": [
            {
                "resolution": 768,
                # subset-level settings override general ones
                "subsets": [{"image_dir": str(tmp_path), "recursive": True}],
            }
        ],
    }
    sanitizer = ConfigSanitizer(True, False, False, True)
    blueprint = BlueprintGenerator(sanitizer).generate(config, argparse.Namespace())
    params = blueprint.dataset_group.datasets[0].subsets[0].params

    assert params.recursive is True
    # the rest of the dataset/subset settings are unaffected
    assert params.caption_extension == ".txt"
    assert params.shuffle_caption is True
    assert params.caption_tag_dropout_rate == 0.5
    assert params.flip_aug is True
    assert blueprint.dataset_group.datasets[0].params.enable_bucket is True


def test_recursive_can_be_overridden_per_subset(tmp_path: Path):
    config = {
        "datasets": [
            {
                "resolution": 768,
                "recursive": True,
                "subsets": [
                    {"image_dir": str(tmp_path / "a")},
                    {"image_dir": str(tmp_path / "b"), "recursive": False},
                ],
            }
        ]
    }
    sanitizer = ConfigSanitizer(True, False, False, True)
    blueprint = BlueprintGenerator(sanitizer).generate(config, argparse.Namespace())
    subsets = blueprint.dataset_group.datasets[0].subsets

    assert subsets[0].params.recursive is True
    assert subsets[1].params.recursive is False
