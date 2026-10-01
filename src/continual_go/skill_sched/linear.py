from jaxtyping import ScalarLike
from typing import Self
from . import SkillScheduler


class LinearSkillScheduler(SkillScheduler):
    value: ScalarLike
    decay: ScalarLike = 1.0
    step: ScalarLike = 0

    def get(self) -> tuple[ScalarLike, Self]:
        return self.value * (self.decay**self.step), self.replace(step=self.step + 1)
