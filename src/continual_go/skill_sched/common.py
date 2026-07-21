from abc import ABC, abstractmethod
from jaxtyping import ScalarLike
from flax.struct import PyTreeNode
from typing import Self


class SkillScheduler(ABC, PyTreeNode):
    """Common interface for skill schedulers."""

    @abstractmethod
    def update(self) -> Self:
        """Runs a single update step in the scheduler."""

    @abstractmethod
    def get(self) -> ScalarLike:
        """Returns the new skill value."""
