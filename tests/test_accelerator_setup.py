import argparse

import pytest
import torch

from library.accelerator_setup import apply_cuda_performance_options


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_allow_tf32_keeps_inductor_precision_getter_usable():
    original_matmul_precision = torch.get_float32_matmul_precision()
    original_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    original_bf16_reduced_precision_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    original_cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        apply_cuda_performance_options(
            argparse.Namespace(cuda_allow_tf32=True, cuda_cudnn_benchmark=False)
        )

        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    finally:
        torch.set_float32_matmul_precision(original_matmul_precision)
        torch.backends.cudnn.allow_tf32 = original_cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_bf16_reduced_precision_reduction
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
