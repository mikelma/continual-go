from jaxtyping import ScalarLike, Array, Float
from typing import Self
import jax.numpy as jnp
from flax import struct
from . import SkillScheduler
from interpax import interp1d


@struct.dataclass
class InterpolatedSkillScheduler(SkillScheduler):
    xs: Float[Array, " steps"]
    ys: Float[Array, " steps"]
    step: int = 0

    def get(self) -> tuple[ScalarLike, Self]:
        value = interp1d(self.step, self.xs, self.ys, method="akima")
        value = jnp.clip(value, 0, 1)
        return value, self.replace(step=self.step + 1)
