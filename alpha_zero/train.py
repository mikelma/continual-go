# modified from: https://github.com/sotetsuk/pgx/blob/main/examples/alphazero/train.py

import datetime
import os
import pickle
import time
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import wandb
from pydantic import BaseModel
import tyro

from continual_go.alpha_zero.network import AZNet
from continual_go.alpha_zero.config import Config
from continual_go import ContinualGo, State

config = tyro.cli(Config)
assert config.selfplay_batch_size * config.max_num_steps % config.training_batch_size == 0, "selfplay_batch_size * max_num_steps must be divisble by training_batch_size"
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
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: State):
    # model: params
    # state: embedding (batched)
    del rng_key
    model_params, model_state = model

    state, reward = jax.vmap(env.step_turn)(state, action)

    # (batch, H, W, 1)
    obs = (state.turn[:, None, None] * state.board / env.k)[..., None]

    (logits, value), _ = forward.apply(model_params, model_state, obs, is_eval=True)

    # full legal-action mask: empty + non-suicide + not-ko
    legal = jax.vmap(env.legal_actions)(state).reshape(logits.shape)
    logits = jnp.where(legal, logits, jnp.finfo(logits.dtype).min)

    # normalize reward to match the tanh-bounded value head
    #TODO(Esraa): not sure about this normalization, could remove the tanh from the value head instead
    reward = reward.astype(value.dtype) / env.k

    discount = -config.gamma * jnp.ones_like(value)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    action_weights: jnp.ndarray


def selfplay(model, state: State, rng_key: jnp.ndarray) -> tuple[SelfplayOutput, State]:
    model_params, model_state = model

    def step_fn(state, key) -> tuple[State, SelfplayOutput]:
        # observation: (batch, H, W, 1) — turn is per-batch scalar, broadcast over H,W.
        observation = (state.turn[:, None, None] * state.board / env.k)[..., None]

        (logits, value), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )

        # full legal-action mask at the MCTS root
        invalid_actions = ~jax.vmap(env.legal_actions)(state).reshape(logits.shape)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        state, reward = jax.vmap(env.step_turn)(state, policy_output.action)

        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=reward,
        )

    key_seq = jax.random.split(rng_key, config.max_num_steps)
    final_state, data = jax.lax.scan(step_fn, state, key_seq)

    return data, final_state


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray


def compute_loss_input(model, data: SelfplayOutput, final_state: State) -> Sample:
    model_params, model_state = model

    # bootstrap V(s_T) from the network for the truncated tail.
    final_obs = (
        final_state.turn[:, None, None] * final_state.board / env.k
    )[..., None]
    (_, v_T), _ = forward.apply(model_params, model_state, final_obs, is_eval=True)
    v_T = jax.lax.stop_gradient(v_T)

    # reverse-accumulate the discounted return with two-player sign flip.
    # v_t = (r_t / k) + (-gamma) * v_{t+1};  init carry = V(s_T).
    discount = -config.gamma

    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix].astype(carry.dtype) / env.k + discount * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        v_T,
        jnp.arange(config.max_num_steps),
    )

    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = jnp.mean(optax.l2_loss(value, samples.value_tgt))

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss

@jax.jit
def experiment_loop(rng_key, model, state, opt_state):
    # Selfplay (continuing — state is carried over from the previous iteration)
    rng_key, subkey = jax.random.split(rng_key)
    data, state = selfplay(model, state, subkey)
    avg_reward = data.reward.mean()
    samples: Sample = compute_loss_input(model, data, state)

    # Flatten (max_num_steps, batch) into a single sample axis, then shuffle.
    samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), samples)
    rng_key, subkey = jax.random.split(rng_key)
    ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
    samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle

    # Make minibatches: (num_updates, training_batch_size, ...)
    num_updates = samples.obs.shape[0] // config.training_batch_size
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape((num_updates, config.training_batch_size) + x.shape[1:]),
        samples,
    )

    def minibatch_train(carry, minibatch):
        model, opt_state = carry
        model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
        return (model, opt_state), (policy_loss, value_loss)

    (model, opt_state), (policy_losses, value_losses) = jax.lax.scan(
        minibatch_train, (model, opt_state), minibatches
    )

    policy_loss = jnp.mean(policy_losses)
    value_loss = jnp.mean(value_losses)

    return state, rng_key, model, opt_state, policy_loss, value_loss, avg_reward


if __name__ == "__main__":
    if not config.wandb:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(project="pgx-az", config=config.model_dump())

    # Initialize model and opt_state
    dummy_state = env.init()
    # (N=1, H, W, C=1): batched single example, one channel for the board state.
    dummy_input = dummy_state.board.reshape(
        1, config.board_size, config.board_size, 1
    ).astype(jnp.float32)
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])

    # persistent env state across iterations (never reset).
    # Shape per leaf: (selfplay_batch_size, ...).
    init_state = env.init()
    state = jax.tree.map(
        lambda x: jnp.broadcast_to(
            jnp.asarray(x)[None],
            (config.selfplay_batch_size,) + jnp.asarray(x).shape,
        ),
        init_state,
    )

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"continual_go_az_{now}_{config.board_size}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        # Store checkpoints
        if iteration % config.save_interval == 0:
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                       "config": config,
                       "rng_key": rng_key,
                       "model": jax.device_get(model),
                       "opt_state": jax.device_get(opt_state),
                       "iteration": iteration,
                       "frames": frames,
                       "hours": hours,
                }
                pickle.dump(dic, f)

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        state, rng_key, model, opt_state, policy_loss, value_loss, avg_reward = experiment_loop(rng_key, model, state, opt_state)
        frames += config.selfplay_batch_size * config.max_num_steps
        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": float(policy_loss),
                "train/value_loss": float(value_loss),
                "train/avg_reward_per_step": float(avg_reward),
                "hours": hours,
                "frames": frames,
            }
        )
