"""Tests for library/adaptive_checkpointing.py and its trainer hook.

The coefficients are derived from the block geometry and validated against step measurements on CUDA and ROCm, the
resident part is measured at the first step, the controller must only apply a change when the chosen count actually
changes, and the whole point of placing the decision in `Block.forward` - outside the compiled block body - is that
flipping it must not create compiled graphs.
"""

import argparse

import pytest
import torch
import torch.nn as nn

from library import adaptive_checkpointing, args as args_util
from library.adaptive_checkpointing import AdaptiveCheckpointing, MemoryModel

# measured on gfx1151: (tokens, skip_blocks) -> peak GiB, MOSS-Image ConvRot INT8 DiT, rank-64 LoRA, no text encoder
MEASURED = {(16384, 0): 13.69, (16384, 20): 71.07, (4096, 42): 39.05}
MOSS_FIXED_GIB = 5.96  # DiT 5.4 + LoRA 0.28 + grads 0.28

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def moss_model(fixed_gib=MOSS_FIXED_GIB):
    """The fitted coefficients, as the controller gets them."""
    return MemoryModel(fixed_gib=fixed_gib, per_1k_tokens_gib=0.50, per_block_1k_tokens_gib=0.179, source="measured")


class MossImage(nn.Module):
    """Stands in for the DiT: the class name selects the measured coefficients, the attributes derive them."""

    def __init__(self, num_layers=42):
        super().__init__()
        self.num_layers = num_layers
        self.dim = 3072
        self.ffn_dim = 8192
        self.w = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))


# region memory model


def test_memory_model_reproduces_the_measurements():
    model = moss_model()
    # the fit is one line through 16384 tokens with 0 and 20 uncheckpointed blocks; the 4096-token point is a check
    assert model.peak_gib(16384, 0) == pytest.approx(MEASURED[(16384, 0)], abs=1.0)
    assert model.peak_gib(16384, 20) == pytest.approx(MEASURED[(16384, 20)], abs=1.0)
    assert model.peak_gib(4096, 42) == pytest.approx(MEASURED[(4096, 42)], abs=2.0)


def test_max_skip_blocks_is_monotone_clamped_and_safe():
    model = moss_model()
    budget, blocks = 111.0, 42
    small = model.max_skip_blocks(2304, budget, blocks)
    large = model.max_skip_blocks(16384, budget, blocks)
    assert small == blocks  # a 768x768 bucket fits without any recomputation
    assert 30 <= large <= 36  # ~8 blocks checkpointed at 2048x2048
    assert 0 <= large < small <= blocks
    for tokens in (1024, 4096, 9216, 12288, 16128, 16384):
        n = model.max_skip_blocks(tokens, budget, blocks)
        assert 0 <= n <= blocks
        if n > 0:  # the chosen value must actually fit, and one more must not (when it is not clamped)
            assert model.peak_gib(tokens, n) <= budget + 1e-9
        if n < blocks:
            assert model.peak_gib(tokens, n + 1) > budget


def test_max_skip_blocks_with_no_room_checkpoints_everything():
    model = moss_model()
    assert model.max_skip_blocks(16384, model.fixed_gib, 42) == 0
    assert model.max_skip_blocks(0, 100.0, 42) == 42


# endregion


# region tokens / coefficients


def test_token_count_uses_the_patch_size():
    latents = torch.zeros(1, 32, 256, 256)  # a 2048^2 image at a VAE stride of 8
    assert adaptive_checkpointing.token_count(latents, 2) == 16384
    assert adaptive_checkpointing.token_count(latents, 4) == 4096
    assert adaptive_checkpointing.token_count(torch.zeros(2, 32, 128, 128), 2) == 2 * 64 * 64
    assert adaptive_checkpointing.token_count(torch.zeros(1, 32, 1, 128, 128), 2) == 64 * 64  # 5D cache


def test_resolve_memory_model_defaults_to_the_measured_table():
    model = MossImage()
    resolved = adaptive_checkpointing.resolve_memory_model(argparse.Namespace(), model, allocated_gib=6.2)
    per_1k, per_block = adaptive_checkpointing.MEMORY_MODELS["MossImage"]
    assert (resolved.per_1k_tokens_gib, resolved.per_block_1k_tokens_gib) == (per_1k, per_block)
    assert resolved.fixed_gib == pytest.approx(6.2)  # measured at the first step, not derived from the parameters
    assert "MossImage" in resolved.source


def test_resolve_memory_model_derives_the_coefficients_for_an_unknown_architecture():
    """Without a table entry the coefficients come from `dim`/`ffn_dim`/`num_blocks`, within 21% of the measurements."""

    class SomethingNew(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim, self.ffn_dim, self.num_layers = 3072, 8192, 42
            self.w = nn.Parameter(torch.zeros(4))

    resolved = adaptive_checkpointing.resolve_memory_model(argparse.Namespace(), SomethingNew(), allocated_gib=3.0)
    # the block geometry: 42 * 4 * 3072 bytes per token checkpointed, 11 * 3072 + 7 * 8192 per token per block
    expected_1k = 42 * 4 * 3072 / 2**20 * adaptive_checkpointing.ARCH_SAFETY
    expected_block = (11 * 3072 + 7 * 8192) / 2**20 * adaptive_checkpointing.ARCH_SAFETY
    assert resolved.per_1k_tokens_gib == pytest.approx(expected_1k)
    assert resolved.per_block_1k_tokens_gib == pytest.approx(expected_block)
    # and the derivation is conservative against the measured MossImage coefficients (it errs high: a fallback should
    # not under-predict)
    measured_1k, measured_block = adaptive_checkpointing.MEMORY_MODELS["MossImage"]
    assert 0.8 <= expected_1k / measured_1k <= 1.4
    assert 0.8 <= expected_block / measured_block <= 1.4


def test_resolve_memory_model_reproduces_the_anima_and_moss_measurements():
    """The shipped coefficients reproduce every step measurement of both models on both backends."""

    class Anima(MossImage):
        def __init__(self):
            super().__init__(num_layers=28)
            self.dim, self.ffn_dim = 2048, 8192

    # (model class, base GiB measured at the first step, {(tokens, skip): peak allocated GiB})
    cases = [
        (Anima, 4.386, {(2304, 0): 5.02, (2304, 7): 6.14, (2304, 14): 7.56, (4096, 0): 5.39, (4096, 7): 7.52,
                        (4096, 14): 10.00, (7744, 0): 6.27, (7744, 7): 10.26, (7744, 14): 14.93, (12544, 0): 7.43,
                        (12544, 7): 13.88, (16384, 0): 8.37}),          # CUDA
        (Anima, 3.19, {(4096, 0): 4.37, (4096, 7): 6.54, (16384, 0): 7.36, (16384, 7): 15.81}),  # ROCm
        (MossImage, 8.15, {(4096, 0): 9.85, (4096, 7): 12.77}),                                  # CUDA
        (MossImage, 6.20, {(4096, 0): 7.95, (4096, 7): 10.84, (4096, 14): 14.20,
                           (16384, 0): 12.92, (16384, 7): 24.44}),                               # ROCm
    ]
    for cls, base, points in cases:
        resolved = adaptive_checkpointing.resolve_memory_model(argparse.Namespace(), cls(), allocated_gib=base)
        for (tokens, skip), peak in points.items():
            predicted = resolved.peak_gib(tokens, skip)
            assert abs(predicted - peak) <= 0.45, (cls.__name__, base, tokens, skip, predicted, peak)


def test_anima_skip_decision_at_16_gib():
    """What a 16 GiB card would choose for the buckets of a 2048^2-capable dataset."""

    class Anima(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim, self.ffn_dim, self.num_layers = 2048, 8192, 28
            self.w = nn.Parameter(torch.zeros(1024, dtype=torch.bfloat16))

    # measured at the first step of a real Anima run: DiT 2.47, LoRA 0.17, grads 0.17, ProdigyPlus state and buffers
    base = 4.39
    resolved = adaptive_checkpointing.resolve_memory_model(argparse.Namespace(), Anima(), allocated_gib=base)
    # 16 GiB card, of which the driver hands out 14.88 GiB; the text encoder (1.2 GiB) is resident at the first step and
    # therefore part of `base`; 1.5 GiB of --gpu_memory_reserved_gib with expandable segments
    budget = 14.88 - 4.39 - 1.2 + 4.39 - 1.5
    counts = {t: resolved.max_skip_blocks(t, budget, 28) for t in (4096, 7744, 16384)}
    assert counts[4096] > counts[7744] > counts[16384] >= 0, counts  # small images recompute less
    assert counts[16384] >= 1, counts  # 2048^2 still has headroom for a block or two
    for tokens, n in counts.items():
        assert resolved.peak_gib(tokens, n) <= budget
        assert n == 28 or resolved.peak_gib(tokens, n + 1) > budget


def test_resolve_memory_model_overrides_and_unknown_architecture():
    class SomethingElse(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(4))

    with pytest.raises(ValueError, match="cannot derive a memory model"):
        adaptive_checkpointing.resolve_memory_model(argparse.Namespace(), SomethingElse())

    model = SomethingElse()
    resolved = adaptive_checkpointing.resolve_memory_model(
        argparse.Namespace(
            adaptive_checkpointing_per_1k_tokens=0.4,
            adaptive_checkpointing_per_block_1k_tokens=0.2,
            adaptive_checkpointing_fixed=7.0,
        ),
        model,
    )
    assert (resolved.fixed_gib, resolved.per_1k_tokens_gib, resolved.per_block_1k_tokens_gib) == (7.0, 0.4, 0.2)

    overridden = adaptive_checkpointing.resolve_memory_model(
        argparse.Namespace(adaptive_checkpointing_per_block_1k_tokens=0.25), MossImage(), allocated_gib=6.0
    )
    assert overridden.per_block_1k_tokens_gib == 0.25
    assert overridden.per_1k_tokens_gib == adaptive_checkpointing.MEMORY_MODELS["MossImage"][0]


def test_parameter_gib_counts_weights_and_trainable_gradients():
    model = nn.Linear(100, 100, dtype=torch.bfloat16)  # 10 000 weights + 100 bias, all trainable
    expected = (10100 * 2 * 3) / 2**30  # weights + grads, both bf16... plus the fp32? no: bf16 weights, bf16 grads
    assert adaptive_checkpointing.parameter_gib(model) == pytest.approx(expected)
    model.requires_grad_(False)
    assert adaptive_checkpointing.parameter_gib(model) == pytest.approx((10100 * 2) / 2**30)


# endregion


# region controller


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, skip_blocks):
        self.calls.append(skip_blocks)


def test_controller_applies_only_when_the_value_changes():
    recorder = Recorder()
    controller = AdaptiveCheckpointing(moss_model(), 42, 111.0, recorder)
    n_small = controller.update(2304)
    assert n_small == 42 and recorder.calls == [42]
    controller.update(2304)
    controller.update(2304)
    assert recorder.calls == [42]  # no redundant flag flips
    n_large = controller.update(16384)
    assert n_large < 42 and recorder.calls == [42, n_large]
    controller.update(16384)
    assert recorder.calls == [42, n_large]
    controller.update(2304)
    assert recorder.calls == [42, n_large, 42]


def test_controller_ignores_an_unknown_token_count():
    recorder = Recorder()
    controller = AdaptiveCheckpointing(moss_model(), 42, 111.0, recorder)
    assert controller.update(None) is None
    assert recorder.calls == []
    controller.update(0)
    assert recorder.calls == []


def test_controller_budget_from_free_memory(monkeypatch):
    monkeypatch.setattr(adaptive_checkpointing, "free_gpu_gib", lambda device=None: 115.0)
    recorder = Recorder()
    controller = AdaptiveCheckpointing.from_args(
        argparse.Namespace(
            adaptive_checkpointing_budget=None, gpu_memory_reserved_gib=4.0, adaptive_checkpointing_fixed=MOSS_FIXED_GIB
        ),
        MossImage(),
        recorder,
    )
    assert controller.budget_gib == pytest.approx(115.0 + MOSS_FIXED_GIB - 4.0)
    assert 36 <= controller.skip_blocks_for(16384) <= 42


def test_controller_budget_adds_back_the_resident_part(monkeypatch):
    # a 32 GiB card: 17.7 GiB free at the first step, 6 GiB of that is resident -> the budget is a cap on peak_gib
    # (which includes the resident part), so the resident part is added back instead of being counted twice
    monkeypatch.setattr(adaptive_checkpointing, "free_gpu_gib", lambda device=None: 17.7)
    monkeypatch.setattr(adaptive_checkpointing, "total_gpu_gib", lambda device=None: 32.0)
    controller = AdaptiveCheckpointing.from_args(
        argparse.Namespace(adaptive_checkpointing_budget=None, gpu_memory_reserved_gib=4.0, adaptive_checkpointing_fixed=6.0),
        MossImage(),
        Recorder(),
    )
    assert controller.budget_gib == pytest.approx(17.7 + 6.0 - 4.0)


def test_the_same_config_adapts_to_a_smaller_card(monkeypatch):
    """The chosen count follows the detected card, so moving to a 32 GiB GPU needs no new coefficients."""
    for total, free, largest in ((124.0, 110.8, 30), (32.0, 17.7, 2)):
        monkeypatch.setattr(adaptive_checkpointing, "free_gpu_gib", lambda device=None, free=free: free)
        monkeypatch.setattr(adaptive_checkpointing, "total_gpu_gib", lambda device=None, total=total: total)
        controller = AdaptiveCheckpointing.from_args(
            argparse.Namespace(
                adaptive_checkpointing_budget=None, gpu_memory_reserved_gib=4.0, adaptive_checkpointing_fixed=MOSS_FIXED_GIB
            ),
            MossImage(),
            Recorder(),
        )
        n = controller.skip_blocks_for(16505)
        assert controller.memory_model.peak_gib(16505, n) <= controller.budget_gib
        if largest <= 2:
            assert n <= 5  # a 32 GiB card checkpoints nearly all blocks for the largest bucket
        else:
            assert n >= largest  # a 124 GiB card keeps most blocks uncheckpointed


def test_warns_when_even_a_fully_checkpointed_batch_does_not_fit(caplog):
    controller = AdaptiveCheckpointing(moss_model(), 42, budget_gib=14.0, apply=Recorder())
    with caplog.at_level("WARNING"):
        n = controller.update(16505)  # fully checkpointed it needs ~16 GiB
    assert n == 0
    assert "even with every block checkpointed" in caplog.text
    controller.update(16505)  # warned once
    assert caplog.text.count("even with every block checkpointed") == 1


def test_controller_requires_a_sane_budget(monkeypatch):
    monkeypatch.setattr(adaptive_checkpointing, "free_gpu_gib", lambda device=None: 100.0)
    with pytest.raises(ValueError, match="below"):
        AdaptiveCheckpointing.from_args(
            argparse.Namespace(
                adaptive_checkpointing_budget=None,
                gpu_memory_reserved_gib=101.0,
                adaptive_checkpointing_fixed=MOSS_FIXED_GIB,
            ),
            MossImage(),
            Recorder(),
        )

    class NeedsBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(4))

    with pytest.raises(ValueError, match="block count"):
        AdaptiveCheckpointing.from_args(
            argparse.Namespace(
                adaptive_checkpointing_budget=100.0,
                adaptive_checkpointing_per_1k_tokens=0.5,
                adaptive_checkpointing_per_block_1k_tokens=0.2,
            ),
            NeedsBlocks(),
            Recorder(),
        )


# endregion


# region arguments


def parse(extra):
    parser = argparse.ArgumentParser()
    args_util.add_training_arguments(parser, support_dreambooth=True)
    args_util.add_gradient_checkpointing_skip_blocks_argument(parser)
    args = parser.parse_args(extra)
    return args


def test_adaptive_arguments_parse_with_defaults():
    args = parse(["--adaptive_checkpointing", "--gradient_checkpointing"])
    assert args.adaptive_checkpointing
    assert args.gpu_memory_reserved_gib == 4.0
    assert args.adaptive_checkpointing_budget is None
    assert args.adaptive_checkpointing_per_block_1k_tokens is None
    args_util.verify_gradient_checkpointing_skip_blocks(args)


def test_adaptive_requires_gradient_checkpointing():
    args = parse(["--adaptive_checkpointing"])
    with pytest.raises(ValueError, match="requires --gradient_checkpointing"):
        args_util.verify_gradient_checkpointing_skip_blocks(args)


def test_adaptive_rejects_torch_compile():
    args = parse(["--adaptive_checkpointing", "--gradient_checkpointing", "--torch_compile"])
    with pytest.raises(ValueError, match="torch_compile"):
        args_util.verify_gradient_checkpointing_skip_blocks(args)


def test_adaptive_overrides_the_fixed_skip_blocks():
    args = parse(["--adaptive_checkpointing", "--gradient_checkpointing", "--gradient_checkpointing_skip_blocks", "20"])
    args_util.verify_gradient_checkpointing_skip_blocks(args)
    assert args.gradient_checkpointing_skip_blocks == 0


def test_adaptive_rejects_negative_coefficients():
    args = parse(["--adaptive_checkpointing", "--gradient_checkpointing", "--adaptive_checkpointing_per_block_1k_tokens", "-1"])
    with pytest.raises(ValueError, match="per_block_1k_tokens"):
        args_util.verify_gradient_checkpointing_skip_blocks(args)


# endregion


# region trainer hook


def test_trainer_hook_builds_the_token_count_from_the_batch():
    import train_network

    trainer = train_network.NetworkTrainer()
    batch = {
        "latents": torch.zeros(1, 32, 256, 256),  # 2048^2 image
        "input_ids_list": [torch.zeros(100, dtype=torch.long), torch.zeros(20, dtype=torch.long)],
    }
    assert trainer.get_checkpointing_token_count(None, None, None, batch) == 16384 + 120

    batch = {"latents": torch.zeros(1, 32, 96, 64), "text_encoder_outputs_list": [torch.zeros(1, 77, 4096)]}
    assert trainer.get_checkpointing_token_count(None, None, None, batch) == 96 * 64 // 4 + 77

    assert trainer.get_checkpointing_token_count(None, None, None, {}) is None


def test_trainer_hook_is_inert_without_the_flag():
    import train_network

    trainer = train_network.NetworkTrainer()
    args = argparse.Namespace(adaptive_checkpointing=False)
    trainer.on_step_start(args, None, None, None, None, {"latents": torch.zeros(1, 32, 64, 64)}, None, is_train=True)
    assert trainer._adaptive_checkpointing is None


# endregion


# region compile time


class TinyBlock(nn.Module):
    """A block whose body is compiled and whose checkpointing is a flag read outside the compiled function."""

    def __init__(self, dim=32):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(dim))
        self.checkpointed = True

    def _forward(self, x):
        return torch.relu(x @ self.w + self.b)

    def forward(self, x):
        if not (self.training and self.checkpointed):
            return self._forward(x)
        return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)


@cuda_only
def test_flipping_the_flag_does_not_add_compiled_graphs():
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(1)
        return gm.forward

    block = TinyBlock().cuda()
    block._forward = torch.compile(block._forward, backend=backend, fullgraph=True)  # what compile_transformer does
    x = torch.randn(4, 8, 32, device="cuda", requires_grad=True)

    for flag in (True, False, True, False, True):
        block.checkpointed = flag
        out = block(x)
        out.sum().backward()
        block.w.grad = None
        block.b.grad = None

    assert len(graphs) == 1  # the flag lives outside the compiled body, so no recompilation
