"""Defines policy interfaces."""
from typing import Any, Dict, Optional, Tuple
import abc
import numpy as np


class Policy(abc.ABC):
    r"""Implements a policy \pi(a | s)."""

    @abc.abstractmethod
    def act(self, state: Any, hidden_state: Optional[Any] = None, test: bool = False) -> Tuple[int, Any]:
        """Returns an action given the current state.

        Args:
            state (object): current state.

        Returns:
            action (int): action to take.
        """
        raise NotImplementedError()

    @property
    def stats(self) -> Dict[str, Any]:
        """Returns a dict of relevant statistics about the policy."""
        return {}


class RandomPolicy(Policy):
    """Acts uniformly at random on discrete actions."""

    def __init__(self, action_space: Any) -> None:
        """Constructs on a discrete action space.

        Args:
            action_space (spaces.Discrete): action space of the environment.
        """
        self._action_space = action_space

    def act(self, state: Any, hidden_state: Optional[Any] = None, test: bool = False) -> Tuple[int, Any]:
        del state, hidden_state, test

        return np.random.randint(self._action_space.n), None

    def update(self, experience: Any) -> None:
        pass


class ConstantActionPolicy(Policy):
    """Always returns the same action."""

    def __init__(self, action: int) -> None:
        self._action = action

    def act(self, state: Any, hidden_state: Optional[Any] = None, test: bool = False) -> Tuple[int, Any]:
        del state, hidden_state, test

        return self._action, None

    def update(self, experience: Any) -> None:
        pass