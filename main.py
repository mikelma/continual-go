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


def human_action(state):
    s = input("Enter coordinates (i<space>j): ")
    try:
        i = int(s.split(" ")[0])
        j = int(s.split(" ")[1])
    except Exception as _e:
        print("\n*** Parsing error! Try again ***\n")
        return human_action(state)

    return i * state.size + j


def main():
    key = jax.random.key(42)

    env = ContinualGo()
    state = env.init()


    for i in range(100):
        key, _key = jax.random.split(key)

        print(f">>> step={i}")
        plt.cla()
        plot_board(state.board, ax=plt.gca(), show=False)
        plt.pause(0.1)

        action = human_action(state)
        # action = act_randomly(_key, env.legal_actions(state))

        state, reward = env.step(state, action)

        if reward > 0:
            print(">> Capture!")

    plt.show()


if __name__ == "__main__":
    main()
