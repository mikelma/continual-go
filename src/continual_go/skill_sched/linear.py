from jaxtyping import ScalarLike
import jax.numpy as jnp
from typing import Self
from . import SkillScheduler


class LinearSkillScheduler(SkillScheduler):
    value: ScalarLike
    delta: ScalarLike = 0
    step: ScalarLike = 0

    def get(self) -> tuple[ScalarLike, Self]:
        value = jnp.clip(self.value + (self.step * self.delta), 0, 1)
        return value, self.replace(step=self.step + 1)
