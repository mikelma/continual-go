# ContiualGo ⚫⚪: A big world in a small board!

ContinualGO is an environment designed to challenge reinforcement learning algorithms in Big Worlds, see [Javed & Sutton (2024)](https://openreview.net/pdf?id=Sv7DazuCn8).


+ **Try it yourself ✨:** Install the `continual_go` Python package (next section) and run [`play_human.py`](./play_human.py)!

## Getting started

Using [`uv`](https://docs.astral.sh/uv/) you can install the latest version (main branch) of ContinualGo by running the following command inside an `uv` project:

```bash
uv add git+https://github.com/mikelma/small-world.git
```

Now you can start using ContinualGo in your project! For example,

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from continual_go import ContinualGo, plot_board


def act_randomly(key, mask):
    """Ignore observation and choose randomly from legal actions"""
    mask = mask.reshape(-1)
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(key, logits=logits, axis=-1)

env = ContinualGo(size=9, k=32)  # k: max number of stones per player

state = env.init()  # get the initial state (deterministic)

key = jax.random.key(42)

while True:  # wow! this is truly continual!
    plt.cla()
    plot_board(state.board, ax=plt.gca(), show=False)
    plt.pause(0.1)

    key, _key = jax.random.split(key)
    action = act_randomly(_key, env.legal_actions(state))
    state, reward = env.step(state, action)
    print(reward)
```

## API

ContinualGo has a minimalist API inspired by Gymnasium-like environments.

### `continual_go.ContinualGo` class

This is the project's main class, i.e., the environment. Requires two values for initialization, `size` and `k`: the size of the board, and the maximum number of stones allowed per player respectively.

The main methods of the `ContinualGo` class are the following (for more, please check the [source code](./src/continual_go/game.py)):

| Method name     | Arguments      | Return                     | Description                                                                                                                    |
|-----------------|----------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `init`          | None           | `State`                    | Returns the initial state.                                                                                                     |
| `step`          | `action: int`  | `Tuple[State, float]`      | Executes the given action, returning the next state and the reward of the player that made the move.                           |
| `legal_actions` | `state: State` | `Bool[Array, "size size"]` | Returns a boolean matrix of the board's size, where a `True` values indicates a legal move in that `ij` position of the board. |

### Utilities

```python
continual_go.plot_board(board, ax=None, show=True):`
```

Given a board matrix (`state.board`), creates a matplotlib plot of the board's current state.

## License

ContinualGo is distributed under the terms of the GLPv3 license. See [LICENSE](./LICENSE) for more details.
