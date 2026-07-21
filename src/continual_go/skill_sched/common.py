from abc import ABC, abstractmethod
from jaxtyping import ScalarLike
from flax.struct import PyTreeNode
from typing import Self


class SkillScheduler(ABC, PyTreeNode):
    """Common interface for skill schedulers."""

    @abstractmethod
    def get(self) -> tuple[ScalarLike, Self]:
        """Returns the new skill value together with the new state of the scheduler."""
