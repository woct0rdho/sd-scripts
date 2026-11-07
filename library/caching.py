"""Latent and text-encoder output caching helpers.

Functions that move pre-computed VAE latents and text-encoder hidden states
between memory and ``.npz`` files on disk. Used both during the explicit
``cache_latents`` / ``cache_text_encoder_outputs`` preprocessing passes and on
the fly inside ``BaseDataset``.

The leaf image helpers (``load_image`` / ``IMAGE_TRANSFORMS`` /
``trim_and_resize_if_required``) live in ``library.utils``; ``ImageInfo`` is
only needed for type checks and stays behind ``TYPE_CHECKING`` to keep this
module free of any (circular) dependency on ``library.dataset``.
``HIGH_VRAM`` lives in ``library.accelerator_setup``; it has no cycle with this
module so it is imported directly.
"""

import logging
import os
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL

from library import accelerator_setup
from library.device_utils import clean_memory_on_device
from library.utils import IMAGE_TRANSFORMS, load_image, trim_and_resize_if_required

if TYPE_CHECKING:
    from library.dataset import ImageInfo


logger = logging.getLogger(__name__)

# Storage dtype of the cached latents. Independent of the VAE's weight/compute dtype: the values are quantized once, at
# cache time, so a fp32 cache keeps more than a fp32 VAE produced and a bf16 cache keeps exactly a bf16 VAE's output.
# npz has no bfloat16 dtype (numpy has none and ml-dtypes would store a vendor descr that np.load reads back as raw
# `|V2` bytes), so bf16 is stored as the raw `uint16` bit pattern - the top half of the float32 - which every numpy
# version reads, and the loader reinterprets it as a bfloat16 tensor without copying. fp16 is a native numpy dtype.
LATENT_STORAGE_DTYPES = ("fp32", "fp16", "bf16")
STORAGE_DTYPE_KEY = "latents_dtype"  # str marker inside the npz, so a reader knows what it got

_storage_dtype = "fp32"


def set_storage_dtype(name: str) -> None:
    """Select the dtype for newly written latent caches (and the expected dtype of existing ones)."""
    global _storage_dtype
    if name not in LATENT_STORAGE_DTYPES:
        raise ValueError(f"unknown latent cache dtype {name!r}, expected one of {LATENT_STORAGE_DTYPES}")
    if name != _storage_dtype:
        logger.info(f"latent cache storage dtype: {_storage_dtype} -> {name}")
    _storage_dtype = name


def get_storage_dtype() -> str:
    return _storage_dtype


def storage_np_dtype(name: str = None) -> np.dtype:
    """The numpy dtype a cache in `name` uses on disk (`bf16` is stored as raw uint16 bits)."""
    return {"fp32": np.dtype(np.float32), "fp16": np.dtype(np.float16), "bf16": np.dtype(np.uint16)}[name or _storage_dtype]


def latent_to_storage(tensor: torch.Tensor, name: str = None) -> np.ndarray:
    """A float tensor as it is stored in the npz (`fp32`, `fp16`, or `bf16` as raw uint16 bits).

    `fp32` returns a view when the tensor is already a contiguous float32 CPU tensor (no copy); `bf16` rounds with the
    same round-to-nearest-even conversion torch uses everywhere else.
    """
    name = name or _storage_dtype
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        tensor = tensor.to(device="cpu", dtype=torch.float32)
    if name == "fp32":
        return tensor.numpy()
    if name == "fp16":
        return tensor.to(torch.float16).numpy()
    return tensor.to(torch.bfloat16).view(torch.uint16).numpy()


def latent_from_storage(array: np.ndarray) -> torch.Tensor:
    """A stored latent array as a tensor, without copying: uint16 bits are viewed as bfloat16 in place.

    The numpy array must own its buffer (an `np.load` result does), because the tensor shares it.
    """
    tensor = torch.from_numpy(array)
    if tensor.dtype == torch.uint16:  # bf16 stored as its top 16 bits
        return tensor.view(torch.bfloat16)
    return tensor


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool, alpha_mask: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_resoはWxHなので注意

    if not os.path.exists(npz_path):
        return False

    try:
        npz = np.load(npz_path)
        if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
            return False
        if npz["latents"].shape[1:3] != expected_latents_size:
            return False

        if flip_aug:
            if "latents_flipped" not in npz:
                return False
            if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                return False

        if alpha_mask:
            if "alpha_mask" not in npz:
                return False
            if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:  # HxW => WxH != reso
                return False
        else:
            if "alpha_mask" in npz:
                return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List["ImageInfo"], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        image = (
            load_image(info.absolute_path, use_alpha_mask)
            if info.image is None
            else np.array(info.image, np.uint8)
        )
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae: AutoencoderKL,
    cache_to_disk: bool,
    image_infos: List["ImageInfo"],
    flip_aug: bool,
    use_alpha_mask: bool,
    random_crop: bool,
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        image = (
            load_image(info.absolute_path, use_alpha_mask)
            if info.image is None
            else np.array(info.image, np.uint8)
        )
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(image_infos, latents, flipped_latents, alpha_masks):
        # check NaN
        if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

        if cache_to_disk:
            # save_latents_to_disk(
            #     info.latents_npz,
            #     latent,
            #     info.latents_original_size,
            #     info.latents_crop_ltrb,
            #     flipped_latent,
            #     alpha_mask,
            # )
            pass
        else:
            info.latents = latent
            if flip_aug:
                info.latents_flipped = flipped_latent
            info.alpha_mask = alpha_mask

    if not accelerator_setup.HIGH_VRAM:
        clean_memory_on_device(vae.device)
