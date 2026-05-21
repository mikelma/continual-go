# modified from: https://github.com/sotetsuk/pgx/blob/main/examples/alphazero/train.py

import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import wandb
from pydantic import BaseModel
import tyro

from network import AZNet
from config import Config
from continual_go import ContinualGo, State

devices = jax.local_devices()
num_devices = len(devices)

config = tyro.cli(Config)
env = ContinualGo(size=config.board_size, k=config.max_stones, opponent_model=None, az_config=None)

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


@jax.pmap
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


@jax.pmap
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


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


# @jax.pmap
# def evaluate(rng_key, my_model):
#     """A simplified evaluation by sampling. Only for debugging.
#     Please use MCTS and run tournaments for serious evaluation."""
#     my_player = 0
#     my_model_params, my_model_state = my_model

#     key, subkey = jax.random.split(rng_key)
#     batch_size = config.selfplay_batch_size // num_devices
#     keys = jax.random.split(subkey, batch_size)
#     state = jax.vmap(env.init)(keys)

#     def body_fn(val):
#         key, state, R = val
#         (my_logits, _), _ = forward.apply(
#             my_model_params, my_model_state, state.observation, is_eval=True
#         )
#         opp_logits, _ = baseline(state.observation)
#         is_my_turn = (state.current_player == my_player).reshape((-1, 1))
#         logits = jnp.where(is_my_turn, my_logits, opp_logits)
#         key, subkey = jax.random.split(key)
#         action = jax.random.categorical(subkey, logits, axis=-1)
#         state = jax.vmap(env.step_turn)(state, action)
#         R = R + state.rewards[jnp.arange(batch_size), my_player]
#         return (key, state, R)

#     _, _, R = jax.lax.while_loop(
#         lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
#     )
#     return R


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
    # replicates to all devices (drop-in for the deprecated jax.device_put_replicated)
    # We just add a leading device axis; the @jax.pmap'd functions below handle placement.
    def _replicate(x):
        return jnp.broadcast_to(x[None], (num_devices,) + x.shape)

    model, opt_state = jax.tree.map(_replicate, (model, opt_state))

    # persistent env state across iterations (never reset).
    # Shape per leaf: (num_devices, batch_per_device, ...).
    batch_per_device = config.selfplay_batch_size // num_devices
    init_state = env.init()
    state = jax.tree.map(
        lambda x: jnp.broadcast_to(
            jnp.asarray(x)[None, None],
            (num_devices, batch_per_device) + jnp.asarray(x).shape,
        ),
        init_state,
    )

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"continual_go_az_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize logging dict
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        # if iteration % config.eval_interval == 0:
        #     # Evaluation
        #     rng_key, subkey = jax.random.split(rng_key)
        #     keys = jax.random.split(subkey, num_devices)
        #     R = evaluate(keys, model)
        #     log.update(
        #         {
        #             f"eval/vs_baseline/avg_R": R.mean().item(),
        #             f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
        #             f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
        #             f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
        #         }
        #     )

        # Store checkpoints
        if iteration % config.save_interval == 0:
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            with open(os.path.join(ckpt_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                       "config": config,
                       "rng_key": rng_key,
                       "model": jax.device_get(model_0),
                       "opt_state": jax.device_get(opt_state_0),
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

        # Selfplay (continuing — state is carried over from the previous iteration)
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data, state = selfplay(model, state, keys)
        avg_reward = float(jax.device_get(data.reward).mean())
        samples: Sample = compute_loss_input(model, data, state)

        # Shuffle samples and make minibatches
        samples = jax.device_get(samples)  # (#devices, batch, max_num_steps, ...)
        frames += samples.obs.shape[0] * samples.obs.shape[1] * samples.obs.shape[2]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[3:])), samples)
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)  # shuffle
        num_updates = samples.obs.shape[0] // config.training_batch_size
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), samples
        )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss,
                "train/avg_reward_per_step": avg_reward,
                "hours": hours,
                "frames": frames,
            }
        )
