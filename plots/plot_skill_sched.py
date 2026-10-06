from __future__ import annotations

import jax
import jax.numpy as jnp

from src.continual_go.skill_sched.linear import LinearSkillScheduler
from src.continual_go.skill_sched.harmonic import HarmonicSkillScheduler
import tyro
import matplotlib.pyplot as plt

import dataclasses


@dataclasses.dataclass
class Linear:
    value: float = 0
    delta: float = 0


@dataclasses.dataclass
class Harmonic:
    periods: tuple[float, ...] = (10.0, 50.0, 200.0)
    amps: tuple[float, ...] = (1.0,)
    mul_amp: float = 1


@dataclasses.dataclass
class Args:
    method: Linear | Harmonic
    steps: int = 1_000_000
    seed: int = 42


if __name__ == "__main__":
    args = tyro.cli(Args)
    key = jax.random.key(args.seed)

    clss = [(Linear, LinearSkillScheduler), (Harmonic, HarmonicSkillScheduler)]
    scheduler = None
    for cls_arg, cls_sched in clss:
        if isinstance(args.method, cls_arg):
            cfg = dataclasses.asdict(args.method)

            for k, v in cfg.items():
                print(k, v)
                if isinstance(v, tuple):
                    cfg[k] = jnp.array(v)

            # if "periods" in cfg:
            #     cfg["periods"] = jnp.array(cfg["periods"])

            if "key" in cls_sched.__init__.__code__.co_varnames:
                cfg = dict(
                    {
                        "key": key,
                    },
                    **cfg,
                )
            scheduler = cls_sched(**cfg)
            break

    def _body(sched, _val):
        value, new_sched = sched.get()
        return new_sched, value

    iters = jax.numpy.arange(args.steps)
    _carry, values = jax.lax.scan(_body, scheduler, iters)

    plt.plot(values)
    plt.xlabel("Steps")
    plt.ylabel("Skill value")
    plt.show()
