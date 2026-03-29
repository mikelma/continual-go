import jax
import jax.numpy as jnp
from src.continual_go import ContinualGo, plot_board
from pprint import pprint
import matplotlib.pyplot as plt


def act_randomly(key, mask):
    """Ignore observation and choose randomly from legal actions"""
    mask = mask.reshape(-1)
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(key, logits=logits, axis=-1)


def main():
    key = jax.random.key(42)

    env = ContinualGo()
    state = env.init()

    for i in range(100):
        key, _key = jax.random.split(key)
        action = act_randomly(_key, env.legal_actions(state))
        state, reward = env.step(state, action)

        plt.cla()
        plot_board(state.board, ax=plt.gca(), show=False)
        if reward > 0:
            print(">> Capture!")
            plt.title("Capture!")
        else:
            plt.pause(0.1)

    # coords = jnp.stack(jnp.indices((state.size, state.size)), axis=-1).reshape(-1, 2)
    # liberties = jax.vmap(env.count_liberties, in_axes=(None, 0, 0))(state.board, coords[:, 0], coords[:, 1])
    # print(liberties.reshape((state.size, state.size)))

    plt.show()


if __name__ == "__main__":
    main()
