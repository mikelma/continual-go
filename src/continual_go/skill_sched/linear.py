from jaxtyping import ScalarLike
from typing import Self
from . import SkillScheduler


class LinearSkillScheduler(SkillScheduler):
    value: ScalarLike

    def update(self) -> Self:
        return self

    def get(self) -> ScalarLike:
        return self.value
