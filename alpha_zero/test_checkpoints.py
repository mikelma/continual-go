import jax
import tyro
import haiku as hk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import pickle
import jax.numpy as jnp
from pydantic import BaseModel
import mctx
import uuid
import os
from functools import partial
from typing import Literal, TypeAlias
from continual_go import ContinualGo, State
from continual_go.render import plot_board

# from continual_go.alpha_zero.config import Config
from continual_go.alpha_zero.network import AZNet


SamplingMethod: TypeAlias = Literal[
    "dirichlet-argmax",
    "dirichlet-sample",
    "ranking",
    "ranking-prior",
    "epsilon",
    "epsilon-ranking",
    "clip-epsilon",
    "temperature",
    "epsilon-ranking-prior",
    "default",
]


class Args(BaseModel):
    load_path_a: str = (
        "checkpoints/continual_go_trained_az_with_legal_actions/000025.ckpt"
    )
    load_path_b: str = (
        "checkpoints/continual_go_trained_az_with_legal_actions/000400.ckpt"
    )
    video_path: str = "game.gif"

    seed: int = 42

    skill_level: float = 1.0
    dirichlet_alpha: float = 1.0
    rank_var_mul: float = 3.0
    eps_margin: float | tuple[float, float] = jnp.inf

    board_size: int = 9
    max_stones: int = 32

    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True

    gamma: float = 0.99

    num_simulations_a: int = 32
    num_simulations_b: int = 32
    gumbel_a: float = 0
    gumbel_b: float = 0
    max_num_steps: int = 256

    sampling_method: SamplingMethod = "default"

    record_gif: bool = False
    show_plot: bool = False
    font_size: int = 16
    save_csv: bool = False
    csv_dir: str = "./"


config = tyro.cli(Args)
env = ContinualGo.create_selfplay(size=config.board_size, k=config.max_stones)


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: State):
    # model: params
    # state: embedding (batched)
    del rng_key
    model_params, model_state = model

    state, reward = jax.vmap(env.step_turn)(state, action)

    # (batch, H, W, 1)
    obs = (state.turn[:, None, None] * state.board / env.k)[..., None]

    (logits, value), _ = forward.apply(model_params, model_state, obs, is_eval=True)

    # legal-action mask: opponent can play any empty cell
    occupancy_free = (state.board == 0).reshape(logits.shape)
    logits = jnp.where(occupancy_free, logits, jnp.finfo(logits.dtype).min)

    # normalize reward to match the tanh-bounded value head
    # TODO(Esraa): not sure about this normalization, could remove the tanh from the value head instead
    reward = reward.astype(value.dtype) / env.k

    discount = -config.gamma * jnp.ones_like(value)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


def temperature_sampling(
    key: jax.Array, prior: jax.Array, skill_level: float, legal_mask: jax.Array
):
    logits = jnp.log(prior + 1e-8)
    scaled_logits = skill_level * logits
    scaled_logits = jnp.where(legal_mask, scaled_logits, -jnp.inf)
    action = jax.random.categorical(key, scaled_logits)
    return jnp.expand_dims(action, 0)


def ranking_based_sampling(
    key: jax.Array,
    prior: jax.Array,
    num_actions: int,
    skill_level: float,
    legal_mask: jax.Array,
    use_prior: bool = False,
):
    ordering = jnp.argsort(prior)
    ranking = jnp.empty_like(prior)
    ranking = ranking.at[ordering].set(jnp.arange(num_actions))

    logits = skill_level * ranking

    if use_prior:
        logits = logits + jnp.log(prior + 1e-8)

    logits = jnp.where(legal_mask, logits, -jnp.inf)

    action = jax.random.categorical(key, logits)

    return jnp.expand_dims(action, 0)


def epsilon_sampling(
    key: jnp.ndarray,
    p: jax.Array,
    original_action: jax.Array,
    num_actions: int,
    skill_level: float,
    legal_mask: jax.Array,
):
    key_sample, key_choice = jax.random.split(key)

    # Sample a random but legal action
    legal_logits = jnp.where(legal_mask, 0.0, -jnp.inf)
    random_action = jax.random.categorical(key_sample, legal_logits)

    return jax.lax.select(
        jax.random.uniform(key_choice) >= skill_level,
        random_action,
        jnp.squeeze(original_action),
    )


def epsilon_ranking_sampling(
    key: jnp.ndarray,
    p: jax.Array,
    original_action: jax.Array,
    num_actions: int,
    skill_level: float,
    legal_mask: jax.Array,
    var_mul: float = 3.0,
    use_prior: bool = False,
):
    key_sample, key_choice = jax.random.split(key)

    sampled_action = ranking_based_sampling(
        key=key_sample,
        prior=p,
        num_actions=num_actions,
        skill_level=skill_level * var_mul,
        legal_mask=legal_mask,
        use_prior=use_prior,
    )

    return jax.lax.select(
        jax.random.uniform(key_choice) >= skill_level,
        jnp.squeeze(sampled_action),
        jnp.squeeze(original_action),
    )


@partial(jax.jit, static_argnames=("sampling_method", "num_simulations_b"))
def play(
    model_a,
    model_b,
    state: State,
    rng_key: jnp.ndarray,
    skill_level: float = 1.0,
    dirichlet_alpha: float = 1.0,
    sampling_method: SamplingMethod = "dirichlet-argmax",
    var_mul: float = 3.0,
    eps_q_margin: float = jnp.inf,
    num_simulations_b: int = 32,
):
    state = jax.tree.map(lambda x: x[None], state)

    def step_fn(state, key):
        # Model A's turn
        model_a_params, model_a_state = model_a
        obs_a = (state.turn[:, None, None] * state.board / env.k)[..., None]
        (logits_a, value_a), _ = forward.apply(
            model_a_params, model_a_state, obs_a, is_eval=True
        )
        legal_a = jax.vmap(env.legal_actions)(state)
        root_a = mctx.RootFnOutput(
            prior_logits=logits_a, value=value_a, embedding=state
        )
        key, mctx_key = jax.random.split(key)
        policy_a = mctx.gumbel_muzero_policy(
            params=model_a,
            rng_key=mctx_key,
            root=root_a,
            recurrent_fn=recurrent_fn,  # ty: ignore[invalid-argument-type]
            num_simulations=config.num_simulations_a,
            invalid_actions=(~legal_a).reshape(logits_a.shape),
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=config.gumbel_a,
        )

        key, sample_key = jax.random.split(key)
        if sampling_method == "dirichlet-sample":
            policy_a_action = jax.random.categorical(
                sample_key, policy_a.action_weights
            )
        else:
            policy_a_action = policy_a.action
        state_a, reward_a = jax.vmap(env.step_turn)(state, policy_a_action)

        # Model B's turn
        model_b_params, model_b_state = model_b
        obs_b = (state_a.turn[:, None, None] * state_a.board / env.k)[..., None]
        (logits_b, value_b), _ = forward.apply(
            model_b_params, model_b_state, obs_b, is_eval=True
        )
        legal_b = jax.vmap(env.legal_actions)(state_a)
        root_b = mctx.RootFnOutput(
            prior_logits=logits_b, value=value_b, embedding=state_a
        )
        key, mctx_key, key_noise, key_sample = jax.random.split(key, num=4)
        policy_b = mctx.gumbel_muzero_policy(
            params=model_b,
            rng_key=mctx_key,
            root=root_b,
            recurrent_fn=recurrent_fn,  # ty: ignore[invalid-argument-type]
            num_simulations=num_simulations_b,
            invalid_actions=(~legal_b).reshape(logits_b.shape),
            qtransform=mctx.qtransform_completed_by_mix_value,
            # max_depth=(config.num_simulations * skill_level).astype(jnp.int32),  # ty: ignore
            gumbel_scale=config.gumbel_b,
        )

        # qvalues = policy_b.search_tree.summary().qvalues
        # qvalues = jnp.where(qvalues == 0, jnp.nan, qvalues)
        # jax.debug.print(
        #     "variace={v}, max={max}, min={min}",
        #     # m=jnp.nanmean(qvalues),
        #     v=jnp.nanstd(qvalues),
        #     max=jnp.nanmax(qvalues),
        #     min=jnp.nanmin(qvalues),
        # )

        policy_b_qvalues = policy_b.search_tree.summary().qvalues

        # Decide which action to take based on the policy B output and the sampling method
        if sampling_method == "default":
            policy_b_action = policy_b.action

        elif (
            sampling_method == "dirichlet-argmax"
            or sampling_method == "dirichlet-sample"
        ):
            alphas = jnp.full((env.num_actions,), dirichlet_alpha)
            noise = jax.random.dirichlet(key_noise, alphas)
            noise *= legal_b.reshape(logits_b.shape)
            noise /= noise.sum()
            weights = skill_level * policy_b.action_weights + (1 - skill_level) * noise

            if sampling_method == "dirichlet-sample":
                policy_b_action = jax.random.categorical(key_sample, weights)
            else:
                policy_b_action = jnp.argmax(weights)

        elif sampling_method == "ranking" or sampling_method == "ranking-prior":
            policy_b_action = ranking_based_sampling(
                key=key_sample,
                prior=policy_b.action_weights[0],  # ty: ignore
                num_actions=env.num_actions,
                skill_level=skill_level,
                legal_mask=legal_b.reshape(-1),
                use_prior=sampling_method == "ranking-prior",
            )

        elif sampling_method in ["epsilon", "clip-epsilon"]:
            policy_b_action = epsilon_sampling(
                key=key_sample,
                p=policy_b.action_weights,  # ty: ignore[invalid-argument-type]
                original_action=policy_b.action,  # ty: ignore[invalid-argument-type]
                num_actions=env.num_actions,
                skill_level=skill_level,
                legal_mask=legal_b.reshape(-1),
            )

            if sampling_method == "clip-epsilon":
                policy_b_qvalues = jnp.where(
                    policy_b_qvalues == 0, jnp.nan, policy_b_qvalues
                )
                margin = jnp.squeeze(
                    jnp.nanmax(policy_b_qvalues, axis=-1)
                    - jnp.nanmin(policy_b_qvalues, axis=-1)
                )
                policy_b_action = jax.lax.select(
                    margin >= eps_q_margin,
                    jnp.squeeze(policy_b.action),
                    policy_b_action,
                )

        elif sampling_method == "epsilon-ranking":
            policy_b_action = epsilon_ranking_sampling(
                key=key_sample,
                p=policy_b.action_weights,  # ty: ignore[invalid-argument-type]
                original_action=jnp.squeeze(policy_b.action),
                num_actions=env.num_actions,
                skill_level=skill_level,
                legal_mask=legal_b.reshape(-1),
                var_mul=var_mul,
            )

        elif sampling_method == "temperature":
            policy_b_action = temperature_sampling(
                key=key_sample,
                prior=policy_b.action_weights,  # ty: ignore[invalid-argument-type]
                skill_level=skill_level,
                legal_mask=legal_b.reshape(-1),
            )

        else:
            raise Exception(f"Invalid sampling method '{sampling_method}'")

        policy_b_action = jnp.expand_dims(policy_b_action, 0)
        state_b, reward_b = jax.vmap(env.step_turn)(state_a, policy_b_action)

        return state_b, (
            state_a.board,
            state_b.board,
            reward_a,
            reward_b,
            policy_b_qvalues,
            policy_b.action != policy_b_action,
        )

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

    eps_margin = args.eps_margin
    if isinstance(args.eps_margin, tuple):
        eps_margin = sum(eps_margin) - args.skill_level * eps_margin[1]  # ty: ignore

    state = env.init()
    board_a, board_b, reward_a, reward_b, policy_b_qvalues, action_b_changes = play(
        model_a,
        model_b,
        state,
        key,
        args.skill_level,
        args.dirichlet_alpha,
        sampling_method=args.sampling_method,
        var_mul=args.rank_var_mul,
        eps_q_margin=eps_margin,
        num_simulations_b=config.num_simulations_b,
    )

    label_a = args.load_path_a.split("/")[-1]
    label_b = args.load_path_b.split("/")[-1]

    if args.save_csv:
        fname = f"eval_checkpoints_{uuid.uuid4()}.csv"
        fname = os.path.join(args.csv_dir, fname)
        ret_A = jnp.cumsum(reward_a)[-1]
        ret_B = jnp.cumsum(reward_b)[-1]

        with open(fname, "w") as f:
            f.write(
                "seed,sampling_method,board_size,k,num_steps,skill_level,dirichlet_alpha,model_A,model_B,sims_A,sims_B,gumbel_A,gumbel_B,return_A,return_B\n"
            )
            if args.sampling_method == "epsilon-ranking":
                args.sampling_method += f"-{args.rank_var_mul}"

            if args.sampling_method == "clip-epsilon" and isinstance(
                args.eps_margin, tuple
            ):
                args.sampling_method += f"-{args.eps_margin[0]}-{args.eps_margin[1]}"

            f.write(
                f"{args.seed},{args.sampling_method},{args.board_size},{args.max_stones},{args.max_num_steps},{args.skill_level},{args.dirichlet_alpha},{args.load_path_a},{args.load_path_b},{config.num_simulations_a},{args.num_simulations_b},{config.gumbel_a},{config.gumbel_b},{ret_A},{ret_B}\n"
            )

    if args.show_plot:
        fig = plt.figure()
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1])
        plt.rcParams.update({"font.size": args.font_size})

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        axs = [ax1, ax2, ax3]

        ax1.plot(jnp.cumsum(reward_b), label="AlphaZero B", color="#2980b9")
        ax1.plot(jnp.cumsum(reward_a), label="AlphaZero A", color="#e74c3c")
        ax1.set_ylabel("Cumulative reward", fontsize=args.font_size)
        ax1.set_xlabel("Steps", fontsize=args.font_size)

        policy_b_qvalues = policy_b_qvalues.at[policy_b_qvalues == 0].set(jnp.nan)
        ax2.plot(jnp.nanmean(policy_b_qvalues, axis=-1), label="mean")
        ax2.plot(jnp.nanmax(policy_b_qvalues, axis=-1), label="max")
        ax2.plot(jnp.nanmin(policy_b_qvalues, axis=-1), label="min")
        changes = jnp.arange(args.max_num_steps)
        bad_changes = list(
            changes[(jnp.squeeze(action_b_changes) & jnp.squeeze(reward_a > 0))]
        )
        changes = changes[jnp.squeeze(action_b_changes)]
        for i, coord in enumerate(changes):
            kwa = dict(
                x=coord,
                label="changes",
                color="tab:gray",
                alpha=1 if coord in bad_changes else 0.2,
            )
            if i > 0:
                del kwa["label"]
            [ax.axvline(**kwa) for ax in axs]  # ty: ignore[invalid-argument-type]
        ax2.set_ylabel("Q-values (policy B)", fontsize=args.font_size)
        ax2.set_xlabel("Steps", fontsize=args.font_size)

        margin = jnp.nanmax(policy_b_qvalues, axis=-1) - jnp.nanmin(
            policy_b_qvalues, axis=-1
        )
        ax3.plot(margin)
        ax2.set_ylabel("Max Q - Min Q (B)", fontsize=args.font_size)
        ax2.set_xlabel("Steps", fontsize=args.font_size)

        for ax in axs:
            ax.legend(fontsize=args.font_size, frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # plt.tight_layout()
        plt.savefig("reward_curve.png", dpi=500)
        plt.show()

    if args.record_gif:
        # interleave A and B half-steps: board_a[t] then board_b[t]
        # boards have shape (steps, 1, H, W) — squeeze batch dim
        boards_a = jnp.squeeze(board_a, axis=1)  # (steps, H, W)
        boards_b = jnp.squeeze(board_b, axis=1)
        frames = []
        for t in range(boards_a.shape[0]):
            frames.append((boards_a[t], f"Step {t + 1} — A ({label_a}) just played"))
            frames.append((boards_b[t], f"Step {t + 1} — B ({label_b}) just played"))

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
