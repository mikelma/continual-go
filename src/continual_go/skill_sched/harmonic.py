from jaxtyping import ScalarLike, PRNGKeyArray
from typing import Self
import jax
import jax.numpy as jnp
from flax import struct
from . import SkillScheduler


@struct.dataclass
class HarmonicSkillScheduler(SkillScheduler):
    key: PRNGKeyArray
    periods: jax.Array
    amps: jax.Array
    mul_amp: float = 1
    step: int = 0

    def get(self) -> tuple[ScalarLike, Self]:
        new_key, _key = jax.random.split(self.key)

        omegas = (2 * jnp.pi) / self.periods

        waves = -jnp.cos(omegas * self.step)
        waves *= self.amps
        signal = jnp.sum(waves)

        value = signal / self.periods.shape[0]

        amp = jnp.clip(self.step * self.mul_amp, 0, 1)

        value = amp * (value + 1) / 2

        return value, self.replace(key=new_key, step=self.step + 1)
