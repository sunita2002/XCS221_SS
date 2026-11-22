"""Tensorboard and padding utilities."""
from typing import Any
import os

from torch.utils import tensorboard


class EpisodeAndStepWriter(object):
    """Logs to tensorboard against both episode and number of steps."""

    def __init__(self, log_dir: str) -> None:
        self._episode_writer = tensorboard.SummaryWriter(
                os.path.join(log_dir, "episode"))
        self._step_writer = tensorboard.SummaryWriter(
                os.path.join(log_dir, "step"))

    def add_scalar(self, key: str, value: float, episode: int, step: int) -> None:
        self._episode_writer.add_scalar(key, value, episode)
        self._step_writer.add_scalar(key, value, step)
