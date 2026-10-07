from .game import ContinualGo
from .skill_sched.harmonic import HarmonicSkillScheduler
from .skill_control.temperature import TemperatureSkillControl

from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp


BENCHMARKS = {
    "9x9-k16-1": dict(
        size=9,
        k=16,
        opponent_path="https://zenodo.org/records/23078271/files/az_9x9_k16_1.ckpt",
        skill_sched_class=HarmonicSkillScheduler,
        skill_sched_kwargs=dict(
            periods=jnp.array([126229, 160243, 652033, 356959, 998909]),
            amps=jnp.array([0.1, 0.8, 0.8, 1, 1]),
            mul_amp=0.0000005,
        ),
        skill_control_class=TemperatureSkillControl,
    ),
}


def get_benchmark(name: str, key: PRNGKeyArray, **kwargs):
    cfg = BENCHMARKS[name]

    bench_args = dict(
        size=cfg["size"],
        k=cfg["k"],
        opponent_path=cfg["opponent_path"],
        skill_sched=cfg["skill_sched_class"](key, **cfg["skill_sched_kwargs"]),
        skill_control=cfg["skill_control_class"](),
    )

    game = ContinualGo.create(**dict(bench_args, **kwargs))

    return game
