import jax
import jax.numpy as jnp
from src.continual_go import ContinualGo, plot_board
import matplotlib.pyplot as plt


def act_randomly(key, mask):
    """Ignore observation and choose randomly from legal actions"""
    mask = mask.reshape(-1)
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(key, logits=logits, axis=-1)


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

def rollout(key, env, num_timesteps):
    unif_dist = jnp.full((env.num_actions,), 1 / env.num_actions)

    def _step(carry, _):
        key, state = carry
        key, _key = jax.random.split(key)

        # action = act_randomly(_key, env.legal_actions(state))  # legal mov. check
        # action = act_randomly(_key, jnp.zeros((env.num_actions,)))  # no legal mov. check
        action = env.sample_legal_action(key, state, unif_dist)

        state, reward = env.step(state, action)

        return (key, state), reward

    state = env.init()
    _carry, res = jax.lax.scan(_step, (key, state), length=num_timesteps)
    return res

def main():
    key = jax.random.key(42)

    env = ContinualGo(size=9, k=22)

    #############################
    # import time
    # jit_fn = jax.jit(rollout, static_argnums=(2))
    # start = time.time()
    # jit_fn(key, env, 1000).block_until_ready()
    # print("end 1:", time.time() - start)
    # start = time.time()
    # jit_fn(key, env, 1000).block_until_ready()
    # print("end 2:", time.time() - start)
    # quit()
    #############################

    state = env.init()


    for i in range(1_000_000):
        key, _key = jax.random.split(key)

        n_black, n_white = (state.board < 0).sum(), (state.board > 0).sum()
        print(f">>> step={i}, black={n_black}, white={n_white}")

        plt.cla()
        plot_board(state.board, ax=plt.gca(), show=False)
        plt.pause(0.1)

        # action = human_action(state, env.legal_actions(state))
        # action = jax.jit(act_randomly)(_key, env.legal_actions(state))
        dist = jnp.full((env.num_actions,), 1 / env.num_actions)
        action = env.sample_legal_action(key, state, dist)

        state, reward = jax.jit(env.step)(state, action)

        if reward > 0:
            print(">> Capture!")

    plt.show()


if __name__ == "__main__":
    main()
