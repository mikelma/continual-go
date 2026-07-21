from jaxtyping import ScalarLike
from typing import Self
from . import SkillScheduler


class LinearSkillScheduler(SkillScheduler):
    value: ScalarLike

    def get(self) -> tuple[ScalarLike, Self]:
        return self.value, self
