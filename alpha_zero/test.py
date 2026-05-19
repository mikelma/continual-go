import jax
import tyro
import haiku as hk

import os
import pickle
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from pydantic import BaseModel
import mctx
from continual_go import ContinualGo, State
from config import Config
from network import AZNet


class Args(BaseModel):
    load_path: str

    seed: int = 42

    board_size: int = 9
    max_stones: int = 32

    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True

    gamma: float = 0.99

    # selfplay params
    num_simulations: int = 32
    max_num_steps: int = 256


config = tyro.cli(Args)
env = ContinualGo(size=config.board_size, k=config.max_stones)


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

    state, reward = jax.vmap(env.step)(state, action)

    # (batch, H, W, 1)
    obs = (state.turn[:, None, None] * state.board / env.k)[..., None]

    (logits, value), _ = forward.apply(model_params, model_state, obs, is_eval=True)

    #legal-action mask: opponent can play any empty cell
    occupancy_free = (state.board == 0).reshape(logits.shape)
    logits = jnp.where(occupancy_free, logits, jnp.finfo(logits.dtype).min)

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


def act_randomly(key, mask):
    """Ignore observation and choose randomly from legal actions"""
    mask = mask.reshape(-1)
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(key, logits=logits, axis=-1)


def play(model, state: State, rng_key: jnp.ndarray):
    model_params, model_state = model

    def step_fn(state, key):
        # observation: (batch, H, W, 1) — turn is per-batch scalar, broadcast over H,W.
        observation = (state.turn[:, None, None] * state.board / env.k)[..., None]

        (logits, value), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )

        # occupancy-only legal-action mask at the MCTS root
        legal_actions = env.legal_actions(state)
        invalid_actions = (~legal_actions).reshape(logits.shape)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        key, mctx_key = jax.random.split(key)
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=mctx_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=invalid_actions,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        state_az, reward_az = env.step(state, policy_output.action)

        # Run a random agent as the opponent
        key, key_act = jax.random.split(key)
        mask = env.legal_actions(state_az)
        action = act_randomly(key, mask)

        state_op, reward_op = env.step(state_az, policy_output.action)

        return state, (state_az, state_op, reward_az, reward_op)

    key_seq = jax.random.split(rng_key, config.max_num_steps)
    final_state, data = jax.lax.scan(step_fn, state, key_seq)

    print(data)
    quit()

    return data, final_state


def load_checkpoint(ckpt_path):
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract the individual components
    config = checkpoint["config"]
    rng_key = checkpoint["rng_key"]
    model_params = checkpoint["model"]
    opt_state = checkpoint["opt_state"]
    iteration = checkpoint["iteration"]
    frames = checkpoint["frames"]
    hours = checkpoint["hours"]

    # Move the parameters and optimizer state back to the device
    model_params = jax.device_put(model_params)
    opt_state = jax.device_put(opt_state)

    return {
        "config": config,
        "rng_key": rng_key,
        "model": model_params,
        "opt_state": opt_state,
        "iteration": iteration,
        "frames": frames,
        "hours": hours
    }

if __name__ == "__main__":
    args = tyro.cli(Args)

    key = jax.random.key(args.seed)

    ckpt_data = load_checkpoint(args.load_path)
    params = ckpt_data["model"]

    state = env.init()
    play(params, state, key)
    # print(params)
