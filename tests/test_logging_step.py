"""Tests for the tracker step handling in NetworkTrainer.accelerator_logging.

wandb is a separate run per launch, but its history is keyed by `global_step` instead of wandb's per-run counter, so a
resumed run plots on the same x axis as the run it continues (and validation/epoch logs do not shift it).
"""

import pytest

import train_network


class FakeTracker:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def log(self, values, step=None, **kwargs):
        self.calls.append((dict(values), step, kwargs))

    @property
    def steps(self):
        return [call[1] for call in self.calls]


class FakeAccelerator:
    def __init__(self, *trackers):
        self.trackers = list(trackers)

    def get_tracker(self, name):
        return next(tracker for tracker in self.trackers if tracker.name == name)


@pytest.fixture
def trainer():
    return train_network.NetworkTrainer()


@pytest.fixture
def trackers():
    return FakeTracker("tensorboard"), FakeTracker("wandb"), FakeTracker("csv")


def test_step_logging_keys_every_tracker_by_the_step(trainer, trackers):
    tensorboard, wandb, other = trackers
    accelerator = FakeAccelerator(*trackers)

    trainer.accelerator_logging(accelerator, {"loss": 1.0}, step_value=4001, global_step=4001, epoch=8)

    assert tensorboard.steps == [4001]
    assert other.steps == [4001]
    assert wandb.steps == [4001]
    assert wandb.calls[0][0]["global_step"] == 4001
    assert wandb.calls[0][0]["epoch"] == 8


def test_wandb_is_keyed_by_global_step_while_tensorboard_keeps_its_own_axis(trainer):
    tensorboard, wandb = FakeTracker("tensorboard"), FakeTracker("wandb")
    accelerator = FakeAccelerator(tensorboard, wandb)

    # validation rows are offset on the tensorboard axis so that they do not collide; wandb must stay on global_step
    trainer.accelerator_logging(accelerator, {"loss": 1.0}, step_value=4031, global_step=4000, epoch=8, val_step=31)
    assert tensorboard.steps == [4031]
    assert wandb.steps == [4000]
    assert wandb.calls[0][0]["val_step"] == 31

    # epoch rows are keyed by the epoch number for tensorboard, which is a different (non-monotonic) axis for wandb
    trainer.accelerator_logging(accelerator, {"loss": 1.0}, step_value=8, global_step=4200, epoch=8)
    assert tensorboard.steps == [4031, 8]
    assert wandb.steps == [4000, 4200]


def test_wandb_steps_stay_monotonic_across_validation_and_a_resume(trainer):
    wandb = FakeTracker("wandb")
    accelerator = FakeAccelerator(wandb)

    for step in (1, 2, 3):
        trainer.accelerator_logging(accelerator, {"loss": 1.0}, step_value=step, global_step=step, epoch=1)
    # 500 steps later the run is aborted and resumed: the new run keeps logging on the same axis
    for step in (4001, 4002):
        trainer.accelerator_logging(accelerator, {"loss": 1.0}, step_value=step, global_step=step, epoch=8)
        trainer.accelerator_logging(accelerator, {"val": 1.0}, step_value=step + 4, global_step=step, epoch=8, val_step=4)

    assert wandb.steps == [1, 2, 3, 4001, 4001, 4002, 4002]
    assert wandb.steps == sorted(wandb.steps)
