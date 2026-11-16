from .game import ContinualGo
from .skill_sched.harmonic import HarmonicSkillScheduler
from .skill_sched.interp import InterpolatedSkillScheduler
from .skill_control.temperature import TemperatureSkillControl

from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp


BENCHMARKS = {
    # y=0 until x=3e5, then sudden increase to y=1 and y=0 again for 4e5
    "9x9-k16-A": dict(
        size=9,
        k=16,
        opponent_path="https://zenodo.org/records/23078271/files/az_9x9_k16_1.ckpt",
        model_sha="9c5a0bd78505f9f2b7e7de23807e77ca4ffa269170efe57bb1ea846c9522b855",
        skill_sched_class=InterpolatedSkillScheduler,
        skill_sched_kwargs=dict(
            xs=jnp.array([0, 3e5, 3.1e5, 3.5e5, 3.9e5, 4e5, 5e5]),
            ys=jnp.array([0, 0, 0.2, 1, 0.2, 0, 0]),
        ),
        skill_control_class=TemperatureSkillControl,
    ),
    # y=1 until x=2.5e5, then y=0 for the rest of the steps
    "9x9-k16-B": dict(
        size=9,
        k=16,
        opponent_path="https://zenodo.org/records/23078271/files/az_9x9_k16_1.ckpt",
        model_sha="9c5a0bd78505f9f2b7e7de23807e77ca4ffa269170efe57bb1ea846c9522b855",
        skill_sched_class=InterpolatedSkillScheduler,
        skill_sched_kwargs=dict(
            xs=jnp.array([0, 2.5e5, 2.51e5, 5e5]),
            ys=jnp.array([1, 1, 0, 0]),
        ),
        skill_control_class=TemperatureSkillControl,
    ),
    # increasing skill level in oscillations that go up to y=0.26
    "9x9-k16-C": dict(
        size=9,
        k=16,
        opponent_path="https://zenodo.org/records/23078271/files/az_9x9_k16_1.ckpt",
        model_sha="9c5a0bd78505f9f2b7e7de23807e77ca4ffa269170efe57bb1ea846c9522b855",
        skill_sched_class=HarmonicSkillScheduler,
        skill_sched_kwargs=dict(
            periods=jnp.array([126229, 160243, 652033, 356959, 998909]),
            amps=jnp.array([0.1, 0.8, 0.8, 1, 1]),
            mul_amp=0.0000005,
        ),
        skill_control_class=TemperatureSkillControl,
    ),
}


def get_benchmark(name: str, **kwargs):
    cfg = BENCHMARKS[name]

    bench_args = dict(
        size=cfg["size"],
        k=cfg["k"],
        opponent_path=cfg["opponent_path"],
        model_sha=cfg["model_sha"],
        skill_sched=cfg["skill_sched_class"](**cfg["skill_sched_kwargs"]),
        skill_control=cfg["skill_control_class"](),
    )

    game = ContinualGo.create(**dict(bench_args, **kwargs))

    return game
