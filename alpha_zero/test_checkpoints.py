import jax
import tyro
import haiku as hk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import jax.numpy as jnp
from pydantic import BaseModel
import mctx
from continual_go import ContinualGo, State
from continual_go.render import plot_board
from config import Config
from network import AZNet
from test import recurrent_fn, forward_fn


class Args(BaseModel):
    load_path_a: str = "checkpoints/continual_go_trained_az_with_legal_actions/000025.ckpt"
    load_path_b: str = "checkpoints/continual_go_trained_az_with_legal_actions/000400.ckpt"
    video_path: str = "game.gif"

    seed: int = 42

    board_size: int = 9
    max_stones: int = 32

    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True

    gamma: float = 0.99

    num_simulations: int = 32
    max_num_steps: int = 256


config = tyro.cli(Args)
env = ContinualGo(size=config.board_size, k=config.max_stones)


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))



@jax.jit
def play(model_a, model_b, state: State, rng_key: jnp.ndarray):
    state = jax.tree.map(lambda x: x[None], state)

    def step_fn(state, key):
        # Model A's turn
        model_a_params, model_a_state = model_a
        obs_a = (state.turn[:, None, None] * state.board / env.k)[..., None]
        (logits_a, value_a), _ = forward.apply(model_a_params, model_a_state, obs_a, is_eval=True)
        legal_a = jax.vmap(env.legal_actions)(state)
        root_a = mctx.RootFnOutput(
            prior_logits=logits_a, value=value_a, embedding=state
        )
        key, mctx_key = jax.random.split(key)
        policy_a = mctx.gumbel_muzero_policy(
            params=model_a,
            rng_key=mctx_key,
            root=root_a,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=(~legal_a).reshape(logits_a.shape),
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        state_a, reward_a = jax.vmap(env.step)(state, policy_a.action)

        # Model B's turn
        model_b_params, model_b_state = model_b
        obs_b = (state_a.turn[:, None, None] * state_a.board / env.k)[..., None]
        (logits_b, value_b), _ = forward.apply(model_b_params, model_b_state, obs_b, is_eval=True)
        legal_b = jax.vmap(env.legal_actions)(state_a)
        root_b = mctx.RootFnOutput(
            prior_logits=logits_b, value=value_b, embedding=state_a
        )
        key, mctx_key = jax.random.split(key)
        policy_b = mctx.gumbel_muzero_policy(
            params=model_b,
            rng_key=mctx_key,
            root=root_b,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=(~legal_b).reshape(logits_b.shape),
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        state_b, reward_b = jax.vmap(env.step)(state_a, policy_b.action)

        return state_b, (state_a.board, state_b.board, reward_a, reward_b)

    key_seq = jax.random.split(rng_key, config.max_num_steps)
    final_state, data = jax.lax.scan(step_fn, state, key_seq)

    return data


def load_checkpoint(ckpt_path):
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)

    checkpoint["model"] = jax.device_put(checkpoint["model"])
    return checkpoint


if __name__ == "__main__":
    args = tyro.cli(Args)

    key = jax.random.key(args.seed)

    model_a = load_checkpoint(args.load_path_a)["model"]
    model_b = load_checkpoint(args.load_path_b)["model"]

    state = env.init()
    board_a, board_b, reward_a, reward_b = play(model_a, model_b, state, key)

    label_a = args.load_path_a.split("/")[-1]
    label_b = args.load_path_b.split("/")[-1]

    font_size = 16
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': font_size})
    ax.plot(jnp.cumsum(reward_b), label="AlphaZero@100M frames", color='#2980b9')
    ax.plot(jnp.cumsum(reward_a), label="AlphaZero@6M frames", color='#e74c3c')
    ax.set_ylabel("Cumulative reward", fontsize=font_size)
    ax.set_xlabel("Steps", fontsize=font_size)
    ax.legend(fontsize=font_size,frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=500)
    plt.show()

    # interleave A and B half-steps: board_a[t] then board_b[t]
    # boards have shape (steps, 1, H, W) — squeeze batch dim
    boards_a = jnp.squeeze(board_a, axis=1)  # (steps, H, W)
    boards_b = jnp.squeeze(board_b, axis=1)
    frames = []
    for t in range(boards_a.shape[0]):
        frames.append((boards_a[t], f"Step {t+1} — A ({label_a}) just played"))
        frames.append((boards_b[t], f"Step {t+1} — B ({label_b}) just played"))

    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(i):
        board, title = frames[i]
        ax.clear()
        plot_board(board, ax=ax, show=False)
        ax.set_title(title, fontsize=10)

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=300)

    if args.video_path.endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=3)
    else:
        writer = animation.PillowWriter(fps=3)

    ani.save(args.video_path, writer=writer)
    print(f"Saved video to {args.video_path}")
    plt.close(fig)
