"""ControlNet dataset: paired training images and conditioning (control) images.

Extracted from ``library.train_util`` as part of the dataset-split refactor;
imports the abstract :class:`~library.dataset.BaseDataset` and its
:class:`~library.subset.ControlNetSubset` configuration class.
"""

import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator

from library.dataset import (
    IMAGE_TRANSFORMS,
    BaseDataset,
    glob_images,
    load_image,
)
from library.dreambooth_dataset import DreamBoothDataset
from library.strategy_base import LatentsCachingStrategy
from library.subset import ControlNetSubset, DreamBoothSubset
from library.utils import resize_image, setup_logging, trim_and_resize_if_required

setup_logging()
logger = logging.getLogger(__name__)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        train_inpainting: bool,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str] = None,
        skip_image_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            resolution,
            network_multiplier,
            train_inpainting,
            debug_dataset,
            resize_interpolation,
            skip_image_resolution,
        )

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_cropはControlNetDatasetではサポートされていません"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            train_inpainting,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
            skip_image_resolution,
        )

        self.cache_conditioning_latents = False

        # config_util等から参照される値をいれておく（若干微妙なのでなんとかしたい）
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.resize_interpolation = resize_interpolation
        self.enable_bucket = self.dreambooth_dataset_delegate.enable_bucket
        self.min_bucket_reso = self.dreambooth_dataset_delegate.min_bucket_reso
        self.max_bucket_reso = self.dreambooth_dataset_delegate.max_bucket_reso
        self.bucket_reso_steps = self.dreambooth_dataset_delegate.bucket_reso_steps
        self.bucket_no_upscale = self.dreambooth_dataset_delegate.bucket_no_upscale

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / 制御用画像が見つかりませんでした: {missing_imgs}"
        if len(extra_imgs) > 0:
            logger.warning(f"extra conditioning data for {len(extra_imgs)} images / 余分な制御用画像があります: {extra_imgs}")

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def set_cache_conditioning_latents(self, enabled: bool):
        self.cache_conditioning_latents = enabled

    def set_current_strategies(self):
        super().set_current_strategies()
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

        if not self.cache_conditioning_latents:
            return

        caching_strategy = LatentsCachingStrategy.get_strategy()
        if caching_strategy is None:
            return

        image_infos = [info for info in self.dreambooth_dataset_delegate.image_data.values() if info.cond_img_path is not None]
        if len(image_infos) == 0:
            return

        logger.info("caching conditioning latents with caching strategy.")
        original_values = []
        try:
            for info in image_infos:
                original_values.append(
                    (
                        info,
                        info.absolute_path,
                        info.image,
                        info.latents_npz,
                        info.latents,
                        info.latents_flipped,
                        info.latents_original_size,
                        info.latents_crop_ltrb,
                        info.alpha_mask,
                    )
                )
                info.absolute_path = info.cond_img_path
                info.image = None
                info.latents_npz = None
                info.latents = None
                info.latents_flipped = None
                info.latents_original_size = None
                info.latents_crop_ltrb = None
                info.alpha_mask = None

            self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

            for info in image_infos:
                info.cond_latents_npz = info.latents_npz
                if not caching_strategy.cache_to_disk:
                    info.cond_latents = info.latents
                    info.cond_latents_flipped = info.latents_flipped
        finally:
            for (
                info,
                absolute_path,
                image,
                latents_npz,
                latents,
                latents_flipped,
                latents_original_size,
                latents_crop_ltrb,
                alpha_mask,
            ) in original_values:
                info.absolute_path = absolute_path
                info.image = image
                info.latents_npz = latents_npz
                info.latents = latents
                info.latents_flipped = latents_flipped
                info.latents_original_size = latents_original_size
                info.latents_crop_ltrb = latents_crop_ltrb
                info.alpha_mask = alpha_mask

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def _get_conditioning_latents(self, image_info, flipped: bool):
        if image_info.cond_latents is not None:
            cond_latents = image_info.cond_latents_flipped if flipped and image_info.cond_latents_flipped is not None else image_info.cond_latents
            return cond_latents

        if image_info.cond_latents_npz is None:
            return None

        caching_strategy = self.latents_caching_strategy or LatentsCachingStrategy.get_strategy()
        if caching_strategy is None:
            return None

        latents, _, _, flipped_latents, _ = caching_strategy.load_latents_from_disk(
            image_info.cond_latents_npz, image_info.bucket_reso
        )
        if flipped and flipped_latents is not None:
            latents = flipped_latents
        return torch.FloatTensor(latents)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []
        conditioning_latents = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]

            if self.cache_conditioning_latents:
                cond_latents = self._get_conditioning_latents(image_info, flipped)
                if cond_latents is not None:
                    conditioning_latents.append(cond_latents)

            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / 画像サイズが合いません: {image_info.absolute_path}"

                cond_img, _, _ = trim_and_resize_if_required(
                    False,  # TODO support random crop
                    cond_img,
                    image_info.bucket_reso,
                    image_info.resized_size,
                    resize_interpolation=image_info.resize_interpolation,
                )
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = resize_image(
                        cond_img,
                        cond_img.shape[0],
                        cond_img.shape[1],
                        target_size_hw[1],
                        target_size_hw[0],
                        self.resize_interpolation,
                    )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()
        if len(conditioning_latents) > 0:
            assert len(conditioning_latents) == len(conditioning_images), "conditioning latents are missing for some images"
            example["conditioning_latents"] = torch.stack(conditioning_latents).to(memory_format=torch.contiguous_format).float()

        return example
