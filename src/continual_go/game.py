import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray, Bool
from typing import TypeAlias, Any
import haiku as hk
import pickle
import mctx

from alpha_zero.config import Config
from alpha_zero.network import AZNet

import __main__
__main__.Config = Config

IntLike: TypeAlias = Integer[ScalarLike, ""]


@struct.dataclass
class State:
    # Board geometry & gameplay (unchanged semantics from the original env)
    board: Integer[Array, "size size"]                   # signed age: sign=color, magnitude=age
    turn: IntLike                                         # -1 (black) or +1 (white)

    # PGX-style chain bookkeeping (recomputed at the end of every step)
    chain_id: Integer[Array, "size size"]                 # signed chain id; sign=color, 0=empty
    ko: IntLike                                           # forbidden flat index, -1 if none
    num_pseudo:      Integer[Array, "num_actions"]        # per-chain pseudo-liberty stats
    idx_sum:         Integer[Array, "num_actions"]
    idx_squared_sum: Integer[Array, "num_actions"]


# -------- low-level helpers --------


def _adjacent4(mask: jax.Array) -> jax.Array:
    """For each cell, whether it is adjacent (4-neighborhood) to any True cell."""
    from_up = jnp.pad(mask[:-1, :], ((1, 0), (0, 0)))
    from_down = jnp.pad(mask[1:, :], ((0, 1), (0, 0)))
    from_left = jnp.pad(mask[:, :-1], ((0, 0), (1, 0)))
    from_right = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)))
    return from_up | from_down | from_left | from_right


def _shift(arr, di, dj, fill):
    """Return an array where cell (i, j) contains arr[i+di, j+dj].
    Off-grid neighbors are filled with `fill`."""
    if di == -1:
        return jnp.pad(arr[:-1], ((1, 0), (0, 0)), constant_values=fill)
    if di == 1:
        return jnp.pad(arr[1:], ((0, 1), (0, 0)), constant_values=fill)
    if dj == -1:
        return jnp.pad(arr[:, :-1], ((0, 0), (1, 0)), constant_values=fill)
    return jnp.pad(arr[:, 1:], ((0, 0), (0, 1)), constant_values=fill)


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _compute_chain_ids(signed_board: jax.Array) -> jax.Array:
    """Min-propagation flood-fill over same-color 4-connected components.

    For an occupied cell, `chain_id[i,j] = sign(color) * (canonical_label + 1)`
    where canonical_label is the smallest flat index in the component.
    Empty cells get 0.
    """
    n = signed_board.shape[0]
    occupied = signed_board != 0
    color = signed_board.astype(jnp.int32)
    SENTINEL = jnp.int32(n * n)
    flat_idx = jnp.arange(n * n, dtype=jnp.int32).reshape(n, n)
    init_labels = jnp.where(occupied, flat_idx, SENTINEL)

    def body(carry):
        labels, _ = carry
        cands = []
        for (di, dj) in _DIRS:
            nbr_color = _shift(color, di, dj, fill=jnp.int32(0))
            nbr_labels = _shift(labels, di, dj, fill=SENTINEL)
            cands.append(
                jnp.where((nbr_color == color) & occupied, nbr_labels, SENTINEL)
            )
        new_labels = jnp.minimum(
            jnp.minimum(jnp.minimum(cands[0], cands[1]), jnp.minimum(cands[2], cands[3])),
            labels,
        )
        return new_labels, jnp.any(new_labels != labels)

    labels, _ = jax.lax.while_loop(
        lambda s: s[1], body, (init_labels, jnp.bool_(True))
    )
    return jnp.where(occupied, color * (labels + 1), jnp.int32(0))


def _pseudo_stats(
    chain_id: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-chain pseudo-liberty (num_pseudo, idx_sum, idx_squared_sum).

    Each returned array has shape `(n*n,)`, int32, indexed by `abs(chain_id) - 1`
    for occupied cells. Empty cells route their (zero) contribution to an OOB
    bucket that is discarded.
    """
    n = chain_id.shape[0]
    num_actions = n * n
    empty = chain_id == 0
    flat_idx = jnp.arange(num_actions, dtype=jnp.int32).reshape(n, n)

    c = jnp.zeros((n, n), dtype=jnp.int32)
    s = jnp.zeros((n, n), dtype=jnp.int32)
    q = jnp.zeros((n, n), dtype=jnp.int32)
    for (di, dj) in _DIRS:
        nbr_empty = _shift(empty, di, dj, fill=False).astype(jnp.int32)
        nbr_idx = _shift(flat_idx, di, dj, fill=jnp.int32(0))
        c += nbr_empty
        s += nbr_empty * nbr_idx
        q += nbr_empty * nbr_idx * nbr_idx
    c = jnp.where(empty, 0, c)
    s = jnp.where(empty, 0, s)
    q = jnp.where(empty, 0, q)

    chain_ix = (jnp.abs(chain_id) - 1).reshape(-1)
    chain_ix = jnp.where(chain_ix < 0, num_actions, chain_ix)

    def scatter(x):
        z = jnp.zeros(num_actions + 1, dtype=jnp.int32)
        return z.at[chain_ix].add(x.reshape(-1))

    return scatter(c)[:num_actions], scatter(s)[:num_actions], scatter(q)[:num_actions]


def _adj_ixs(xy: IntLike, n: int) -> jax.Array:
    """Flat indices of the 4 neighbors of `xy`; -1 if off-board."""
    i = xy // n
    j = xy - n * i
    return jnp.stack(
        [
            jnp.where(i > 0, xy - n, -1),
            jnp.where(i < n - 1, xy + n, -1),
            jnp.where(j > 0, xy - 1, -1),
            jnp.where(j < n - 1, xy + 1, -1),
        ]
    )


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


def forward_fn(x, num_actions, cfg):
    net = AZNet(
        num_actions=num_actions,
        num_channels=cfg.num_channels,
        num_blocks=cfg.num_layers,
        resnet_v2=cfg.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=False, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))


# -------- the environment --------


class ContinualGo(PyTreeNode):
    size: int = struct.field(pytree_node=False)
    k: int = struct.field(pytree_node=False)  # max number of stones per player
    opponent_model: Any = struct.field(pytree_node=True)
    az_config: Any = struct.field(pytree_node=True)

    @property
    def num_actions(self) -> IntLike:
        return self.size * self.size

    @classmethod
    def create(cls, size: int, k: int, opponent_path: str):
        ckpt_data = load_checkpoint(opponent_path)
        return cls(
            size=size,
            k=k,
            opponent_model=ckpt_data["model"],
            az_config=ckpt_data["config"],
        )

    @classmethod
    def create_selfplay(cls, size: int, k: int):
        return cls(
            size=size,
            k=k,
            opponent_model=None,
            az_config=None
        )

    def init(self) -> State:
        n = self.size
        return State(
            board=jnp.zeros((n, n), dtype=jnp.int32),
            turn=jnp.int32(-1),
            chain_id=jnp.zeros((n, n), dtype=jnp.int32),
            ko=jnp.int32(-1),
            num_pseudo=jnp.zeros(n * n, dtype=jnp.int32),
            idx_sum=jnp.zeros(n * n, dtype=jnp.int32),
            idx_squared_sum=jnp.zeros(n * n, dtype=jnp.int32),
        )

    def step(self, key, state, action: IntLike) -> tuple[State, ScalarLike]:
        # step function used for planning
        def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: State):
            del rng_key
            model_params, model_state = model

            state, reward = jax.vmap(self.step_turn)(state, action)

            # (batch, H, W, 1)
            obs = (state.turn[:, None, None] * state.board / self.k)[..., None]

            (logits, value), _ = forward.apply(
                model_params, model_state, obs, num_actions=self.num_actions, cfg=self.az_config
            )

            # full legal-action mask: empty + non-suicide + not-ko
            legal = jax.vmap(self.legal_actions)(state).reshape(logits.shape)
            logits = jnp.where(legal, logits, jnp.finfo(logits.dtype).min)

            # normalize reward to match the tanh-bounded value head
            reward = reward.astype(value.dtype) / self.k

            discount = -self.az_config.gamma * jnp.ones_like(value)

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
                )
            return recurrent_fn_output, state

        # execute the player's action
        state, player_reward = self.step_turn(state, action)

        # run AlphaZero
        observation = (state.turn * state.board / self.k)[None, ..., None]

        model_params, model_state = self.opponent_model

        (logits, value), _ = forward.apply(
            model_params, model_state, observation, num_actions=self.num_actions, cfg=self.az_config
        )

        legal_actions = self.legal_actions(state)
        invalid_actions = (~legal_actions).reshape(1, self.num_actions)

        batched_state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=batched_state)

        policy_output = mctx.gumbel_muzero_policy(
            params=self.opponent_model,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.az_config.num_simulations,
            invalid_actions=invalid_actions,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )

        next_state, reward_az = self.step_turn(state, policy_output.action[0])

        return next_state, player_reward - reward_az

    def step_turn(self, state: State, action: IntLike) -> tuple[State, ScalarLike]:
        """Single turn step (useful for self-play)."""
        n = self.size
        action = jnp.minimum(jnp.array(n * n - 1), action)
        i = action // n
        j = action - n * i

        # 1. Age the acting player's stones (magnitude grows), then place the new stone.
        board = state.board
        new_board = jax.lax.cond(
            state.turn == -1,
            lambda: jnp.where(board < 0, board - 1, board),
            lambda: jnp.where(board > 0, board + 1, board),
        )
        new_board = new_board.at[i, j].set(state.turn)

        # 2. Chain IDs + pseudo-stats on the post-placement board.
        signed = jnp.sign(new_board).astype(jnp.int32)
        chain_id = _compute_chain_ids(signed)
        nps, ids, iqs = _pseudo_stats(chain_id)

        # 3. Capture opponent chains with no pseudo-liberty (no liberty at all).
        opponent = -state.turn
        chain_ix = jnp.where(chain_id == 0, 0, jnp.abs(chain_id) - 1)
        captured = (signed == opponent) & (nps[chain_ix] == 0)
        new_board = jnp.where(captured, jnp.int32(0), new_board)
        reward = captured.sum()

        # 4. FIFO eviction: if opponent has > k-1 stones left, drop the oldest.
        def remove_oldest(b):
            flat = b.reshape(-1)
            return jax.lax.cond(
                opponent > 0,
                lambda: flat.at[jnp.argmax(flat)].set(0),
                lambda: flat.at[jnp.argmin(flat)].set(0),
            ).reshape((n, n))

        over = (jnp.sign(new_board) == opponent).sum() > (self.k - 1)
        new_board = jax.lax.cond(
            over, lambda: remove_oldest(new_board), lambda: new_board
        )

        # 5. Recompute chain IDs + stats on the FINAL board.
        # Required because (a) captures freed up empties, and (b) FIFO eviction
        # may have split a chain.
        signed = jnp.sign(new_board).astype(jnp.int32)
        chain_id = _compute_chain_ids(signed)
        nps, ids, iqs = _pseudo_stats(chain_id)

        # 6. Simple PGX-style ko: forbid the captured cell next turn iff
        #    (a) exactly one stone was captured,
        #    (b) the placed stone is a singleton chain (no friendly merge),
        #    (c) the placed stone's chain is in atari.
        placed_id = chain_id[i, j]
        placed_alive = placed_id != 0
        placed_ix = jnp.where(placed_alive, jnp.abs(placed_id) - 1, 0)

        captured_flat = captured.reshape(-1)
        single_capture_idx = jnp.argmax(captured_flat).astype(jnp.int32)
        single_capture = captured_flat.sum() == 1

        placed_in_atari = (ids[placed_ix] ** 2) == (
            iqs[placed_ix] * nps[placed_ix]
        )
        placed_is_singleton = (chain_id == placed_id).sum() == 1

        new_ko = jnp.where(
            single_capture & placed_alive & placed_in_atari & placed_is_singleton,
            single_capture_idx,
            jnp.int32(-1),
        )

        return (
            state.replace(
                board=new_board,
                turn=opponent,
                chain_id=chain_id,
                ko=new_ko,
                num_pseudo=nps,
                idx_sum=ids,
                idx_squared_sum=iqs,
            ),
            reward,
        )

    def count_liberties(
        self, board: Integer[Array, "size size"], i: IntLike, j: IntLike
    ) -> IntLike:
        """Kept for back-compat; not used internally anymore."""
        stone = board[i, j]

        def on_empty(_):
            return jnp.int32(0)

        def on_stone(_):
            color = jnp.sign(stone)
            same_color = jnp.sign(board) == color

            group0 = jnp.zeros(board.shape, dtype=jnp.bool_)
            group0 = group0.at[i, j].set(True)

            def cond_fn(state):
                group, changed = state
                return changed

            def body_fn(state):
                group, _ = state
                expanded = group | (_adjacent4(group) & same_color)
                changed = jnp.any(expanded != group)
                return expanded, changed

            group, _ = jax.lax.while_loop(cond_fn, body_fn, (group0, jnp.bool_(True)))

            liberties_mask = _adjacent4(group) & (board == 0)
            return liberties_mask.sum(dtype=jnp.int32)

        return jax.lax.cond(stone == 0, on_empty, on_stone, operand=None)

    def sample_legal_action(
        self,
        key: PRNGKeyArray,
        state: State,
        weights: Float[Scalar, " {self.size**2}"],
        eps: ScalarLike = 1e-5,
    ) -> IntLike:
        """Categorical sample over the action distribution, with illegal moves masked out."""
        weights = weights + eps
        legal = self.legal_actions(state).reshape(-1)
        weights = jnp.where(legal, weights, jnp.zeros_like(weights))
        probs = weights / weights.sum()
        logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
        return jax.random.categorical(key, logits=logits, axis=-1)

    def legal_actions(self, state: State) -> Bool[Array, "size size"]:
        """O(n^2) legal-action mask using state-resident chain stats. Pure lookup."""
        n = self.size
        is_empty = state.chain_id == 0
        my_sign = state.turn

        chain_ix = jnp.where(is_empty, 0, jnp.abs(state.chain_id) - 1)

        # Cauchy-Schwarz: a chain is in atari iff all its pseudo-libs point at the
        # same empty cell, i.e. idx_sum^2 == idx_squared_sum * num_pseudo.
        in_atari_per_chain = (state.idx_sum ** 2) == (
            state.idx_squared_sum * state.num_pseudo
        )
        cell_in_atari = jnp.where(is_empty, False, in_atari_per_chain[chain_ix])

        has_lib = (state.chain_id * my_sign > 0) & ~cell_in_atari       # mine, >=2 libs
        can_kill = (state.chain_id * (-my_sign) > 0) & cell_in_atari    # opp in atari

        e_flat = is_empty.reshape(-1)
        k_flat = can_kill.reshape(-1)
        h_flat = has_lib.reshape(-1)

        def is_adj_ok(xy):
            adj = _adj_ixs(xy, n)
            on_board = adj >= 0
            idx = jnp.where(on_board, adj, 0)
            return ((e_flat[idx] | k_flat[idx] | h_flat[idx]) & on_board).any()

        mask = e_flat & jax.vmap(is_adj_ok)(jnp.arange(n * n))
        mask = jax.lax.select(state.ko == -1, mask, mask.at[state.ko].set(False))
        return mask.reshape(n, n)
