from typing import Dict, Optional, Tuple

import torch


def alpha_to_float(alpha) -> Optional[float]:
    if alpha is None:
        return None
    if type(alpha) == torch.Tensor:
        return float(alpha.detach().float().cpu().item())
    return float(alpha)


def effective_alpha(alpha, dim: int) -> float:
    alpha = alpha_to_float(alpha)
    return float(dim) if alpha is None or alpha == 0 else alpha


def alpha_tensor(alpha: float, like=None) -> torch.Tensor:
    dtype = torch.float32
    if type(like) == torch.Tensor and like.dtype.is_floating_point:
        dtype = like.dtype
    return torch.tensor(alpha, dtype=dtype)


def extract_lora_dims_and_alphas(weights_sd) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]

    for lora_name, dim in modules_dim.items():
        if lora_name not in modules_alpha:
            modules_alpha[lora_name] = torch.tensor(dim, dtype=torch.float32)

    return modules_dim, modules_alpha


def prepare_lora_state_dict_for_network_alpha(weights_sd, network_alpha: Optional[float]):
    modules_dim, modules_alpha = extract_lora_dims_and_alphas(weights_sd)
    prepared_sd = dict(weights_sd)
    rescaled_modules = 0

    for lora_name, dim in modules_dim.items():
        old_alpha = effective_alpha(modules_alpha.get(lora_name), dim)
        new_alpha = effective_alpha(network_alpha, dim) if network_alpha is not None else old_alpha
        alpha_key = f"{lora_name}.alpha"
        prepared_sd[alpha_key] = alpha_tensor(new_alpha, modules_alpha.get(lora_name))
        modules_alpha[lora_name] = prepared_sd[alpha_key]

        if network_alpha is None or old_alpha == new_alpha:
            continue

        up_key = f"{lora_name}.lora_up.weight"
        if up_key in prepared_sd:
            prepared_sd[up_key] = prepared_sd[up_key] * (old_alpha / new_alpha)
            rescaled_modules += 1

    return prepared_sd, modules_dim, modules_alpha, rescaled_modules


def refresh_lora_module_scales(lora_modules) -> None:
    for lora in lora_modules:
        if not hasattr(lora, "alpha") or not hasattr(lora, "lora_dim"):
            continue
        alpha = effective_alpha(lora.alpha, lora.lora_dim)
        lora.scale = alpha / lora.lora_dim
