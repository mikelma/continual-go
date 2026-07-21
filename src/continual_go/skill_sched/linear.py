from jaxtyping import ScalarLike
from typing import Self
from . import SkillScheduler


class LinearSkillScheduler(SkillScheduler):
    """Always returns the skill value given at initialization."""

    value: ScalarLike

    def update(self) -> Self:
        return self

    def get(self) -> ScalarLike:
        return self.value
