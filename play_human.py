import jax
import jax.numpy as jnp
from src.continual_go import ContinualGo, plot_board
from pprint import pprint
import matplotlib.pyplot as plt


def human_action(state, legal_mask):
    size = state.board.shape[0]
    player = "black" if state.turn == -1 else "white"
    s = input(f"[{player}] Enter coordinates (i j): ")
    try:
        i = int(s.split(" ")[0])
        j = int(s.split(" ")[1])
    except Exception as _e:
        print("\n*** Parsing error! Try again ***\n")
        return human_action(state, legal_mask)

    if not legal_mask[i][j]:
        print("\n*** Illegal move! Try again ***\n")
        return human_action(state, legal_mask)

    return i * size + j



def main():
    key = jax.random.key(42)

    env = ContinualGo(size=9, k=32)
    state = env.init()

    ret_black, ret_white = 0, 0
    for i in range(1_000_000):
        key, _key = jax.random.split(key)

        n_black, n_white = (state.board < 0).sum(), (state.board > 0).sum()

        plt.cla()
        plot_board(state.board, ax=plt.gca(), show=False)
        plt.pause(0.1)

        action = human_action(state, env.legal_actions(state))

        state, reward = jax.jit(env.step)(state, action)

        if state.turn > 0:
            ret_black += reward
        else:
            ret_white += reward

        print(f">>> step={i+1}, n_black={n_black}, n_white={n_white}")
        print(f" >> ret_black={ret_black}, ret_white={ret_white}\n")

    plt.close()


if __name__ == "__main__":
    main()
