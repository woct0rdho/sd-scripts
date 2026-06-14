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


def extract_loha_dims_and_alphas(weights_sd) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        module_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[module_name] = value
        elif key.endswith(".hada_w1_b"):
            modules_dim[module_name] = value.shape[0]

    for module_name, dim in modules_dim.items():
        if module_name not in modules_alpha:
            modules_alpha[module_name] = torch.tensor(dim, dtype=torch.float32)

    return modules_dim, modules_alpha


def prepare_loha_state_dict_for_network_alpha(weights_sd, network_alpha: Optional[float]):
    modules_dim, modules_alpha = extract_loha_dims_and_alphas(weights_sd)
    prepared_sd = dict(weights_sd)
    rescaled_modules = 0

    for module_name, dim in modules_dim.items():
        old_alpha = effective_alpha(modules_alpha.get(module_name), dim)
        new_alpha = effective_alpha(network_alpha, dim) if network_alpha is not None else old_alpha
        alpha_key = f"{module_name}.alpha"
        prepared_sd[alpha_key] = alpha_tensor(new_alpha, modules_alpha.get(module_name))
        modules_alpha[module_name] = prepared_sd[alpha_key]

        if network_alpha is None or old_alpha == new_alpha:
            continue

        scale_key = f"{module_name}.hada_w2_a"
        if scale_key in prepared_sd:
            prepared_sd[scale_key] = prepared_sd[scale_key] * (old_alpha / new_alpha)
            rescaled_modules += 1

    return prepared_sd, modules_dim, modules_alpha, rescaled_modules


def extract_lokr_dims_and_alphas(weights_sd):
    modules_dim = {}
    modules_alpha = {}
    scale_dims = {}
    direct_w2_modules = {}
    w1_decomposed_modules = {}

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        module_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[module_name] = value
        elif key.endswith(".lokr_w1_b"):
            scale_dims[module_name] = value.shape[0]
            w1_decomposed_modules[module_name] = True
        elif key.endswith(".lokr_w2_b"):
            modules_dim[module_name] = value.shape[0]
            scale_dims[module_name] = value.shape[0]
            direct_w2_modules[module_name] = False
        elif key.endswith(".lokr_w2") and module_name not in modules_dim:
            modules_dim[module_name] = max(value.shape[0], value.shape[1])
            direct_w2_modules[module_name] = True

    for module_name, dim in modules_dim.items():
        if module_name not in modules_alpha:
            modules_alpha[module_name] = torch.tensor(scale_dims.get(module_name, dim), dtype=torch.float32)

    return modules_dim, modules_alpha, scale_dims, direct_w2_modules, w1_decomposed_modules


def prepare_lokr_state_dict_for_network_alpha(weights_sd, network_alpha: Optional[float]):
    modules_dim, modules_alpha, scale_dims, direct_w2_modules, w1_decomposed_modules = extract_lokr_dims_and_alphas(weights_sd)
    prepared_sd = dict(weights_sd)
    rescaled_modules = 0

    for module_name in w1_decomposed_modules.keys():
        w1_key = f"{module_name}.lokr_w1"
        w1a_key = f"{module_name}.lokr_w1_a"
        w1b_key = f"{module_name}.lokr_w1_b"
        if w1_key not in prepared_sd and w1a_key in prepared_sd and w1b_key in prepared_sd:
            prepared_sd[w1_key] = prepared_sd[w1a_key] @ prepared_sd[w1b_key]
        prepared_sd.pop(w1a_key, None)
        prepared_sd.pop(w1b_key, None)

    for module_name, dim in modules_dim.items():
        scale_dim = scale_dims.get(module_name)
        direct_w2 = direct_w2_modules.get(module_name, False)
        old_alpha = effective_alpha(modules_alpha.get(module_name), scale_dim or dim)

        if scale_dim is None:
            new_alpha = float(dim)
        elif direct_w2:
            new_alpha = float(dim)
        else:
            new_alpha = effective_alpha(network_alpha, scale_dim) if network_alpha is not None else old_alpha

        alpha_key = f"{module_name}.alpha"
        prepared_sd[alpha_key] = alpha_tensor(new_alpha, modules_alpha.get(module_name))
        modules_alpha[module_name] = prepared_sd[alpha_key]

        scale_key = f"{module_name}.lokr_w1"
        if scale_key not in prepared_sd:
            continue

        if scale_dim is None:
            continue

        if direct_w2:
            scale = old_alpha / scale_dim
        elif old_alpha != new_alpha:
            scale = old_alpha / new_alpha
        else:
            continue

        if scale != 1:
            prepared_sd[scale_key] = prepared_sd[scale_key] * scale
            rescaled_modules += 1

    return prepared_sd, modules_dim, modules_alpha, rescaled_modules


def refresh_lora_module_scales(lora_modules) -> None:
    for lora in lora_modules:
        if not hasattr(lora, "alpha") or not hasattr(lora, "lora_dim"):
            continue
        alpha = effective_alpha(lora.alpha, lora.lora_dim)
        lora.scale = alpha / lora.lora_dim
