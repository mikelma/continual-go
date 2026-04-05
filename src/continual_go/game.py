import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray, Bool
from typing import TypeAlias


IntLike: TypeAlias = Integer[ScalarLike, ""]


@struct.dataclass
class State:
    board: Integer[Array, "size size"]
    turn: IntLike  # -1 (black) or +1 (white)
    prev_boards: Integer[Array, "2 size size"]  # [board_{t-2}, board_{t-1}]


def _adjacent4(mask: jax.Array) -> jax.Array:
    """For each cell, whether it is adjacent (4-neighborhood) to any True cell."""
    from_up = jnp.pad(mask[:-1, :], ((1, 0), (0, 0)))
    from_down = jnp.pad(mask[1:, :], ((0, 1), (0, 0)))
    from_left = jnp.pad(mask[:, :-1], ((0, 0), (1, 0)))
    from_right = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)))
    return from_up | from_down | from_left | from_right


class ContinualGo(PyTreeNode):
    size: int = struct.field(pytree_node=False)
    k: int = struct.field(pytree_node=False)  # max number of stones per player

    @property
    def num_actions(self) -> IntLike:
        return self.size * self.size

    def init(self) -> State:
        board = jnp.zeros((self.size, self.size), dtype=int)
        prev_boards = jnp.zeros((2, self.size, self.size), dtype=int)
        return State(
            board=board, turn=-1, prev_boards=prev_boards
        )

    def step(self, state: State, action: IntLike) -> tuple[State, ScalarLike]:
        n = self.size
        action = jnp.minimum(jnp.array((n * n - 1)), action)  # no pass allowed

        # update board history
        prev_boards = state.prev_boards.at[0].set(state.prev_boards[1])
        prev_boards = prev_boards.at[1].set(state.board)

        # update stone time counts
        board = state.board
        new_board = jax.lax.cond(
            state.turn == -1,
            lambda: jnp.where(board < 0, board - 1, board),
            lambda: jnp.where(board > 0, board + 1, board),
        )

        # set the new stone according to action
        i = action // n
        j = action - n * i
        new_board = new_board.at[i, j].set(state.turn)

        # count the liberties of each stone
        # TODO optimize: only count opponent's liberties, not whole board
        coords = jnp.stack(jnp.indices((self.size, self.size)), axis=-1).reshape(
            -1, 2
        )
        liberties = jax.vmap(self.count_liberties, in_axes=(None, 0, 0))(
            new_board, coords[:, 0], coords[:, 1]
        )
        liberties = liberties.reshape((n, n))

        # check captured stones (& update board): opponent's stones with 0 liberties
        captured = (jnp.sign(new_board) == (-1 * state.turn)) & (liberties == 0)
        new_board = jnp.where(captured, jnp.zeros_like(board, dtype=int), new_board)

        reward = captured.sum()

        def remove_stone(player):
            fb = new_board.reshape(-1)
            return jax.lax.cond(
                player > 0,
                lambda: fb.at[jnp.argmax(fb)].set(0),
                lambda: fb.at[jnp.argmin(fb)].set(0),
            ).reshape((n, n))

        # check if the opponent has more than K-1 stones, and remove the oldest in that case
        opponent = -1 * state.turn
        new_board = jax.lax.cond(
            (jnp.sign(new_board) == opponent).sum() > (self.k - 1),
            lambda: remove_stone(opponent),
            lambda: new_board,
        )

        return state.replace(
            board=new_board,
            turn=opponent,
            prev_boards=prev_boards,
        ), reward

    def count_liberties(
        self, board: Integer[Array, "size size"], i: IntLike, j: IntLike
    ) -> IntLike:
        stone = board[i, j]

        def on_empty(_):
            return jnp.int32(0)

        def on_stone(_):
            color = jnp.sign(stone)  # +1 or -1
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
            eps: ScalarLike = 1e-5
    ) -> IntLike:
        weights += eps

        def normalize_and_sample(key, weights, mask):
            # weights = weights.at[mask].set(0.)
            weights = jnp.where(mask, jnp.zeros_like(weights), weights)
            probs = weights / weights.sum()
            logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
            return jax.random.categorical(key, logits=logits, axis=-1)


        def cond(carry):
            _, action, mask = carry
            i = action // self.size
            j = action - self.size * i
            # simulate playing this action
            new_state, _rwd = self.step(state, action)
            # liberties of the newly played stone
            played_libs = self.count_liberties(new_state.board, i, j)
            suicide = played_libs == 0
            # check if new state is repeats board history
            repeats_board = (jnp.sign(state.prev_boards[1]) == jnp.sign(new_state.board)).all()
            return repeats_board | suicide  # True if action is illegal

        def try_again(carry):
            key, failed_action, mask = carry
            mask = mask.at[failed_action].set(False)
            key, _key = jax.random.split(key)
            action = normalize_and_sample(_key, weights, mask)
            return (key, action, mask)


        key_init, key_loop = jax.random.split(key)
        # set illegal moves (now, occupied positions) to True
        init_mask = (state.board != 0).reshape(-1)
        # sample first action candidate
        action = normalize_and_sample(key_init, weights, init_mask)

        init_carry = (key_loop, action, init_mask)
        carry = jax.lax.while_loop(cond, try_again, init_carry)
        _, action, _ = carry
        return action


    def legal_actions(self, state: State) -> Bool[Array, "size size"]:
        board = state.board
        n = self.size

        def _played_check(ij):
            i, j = ij[0], ij[1]
            action = i * self.size + j
            new_state, _rwd = self.step(state, action)

            # liberties of the newly played stone
            played_libs = self.count_liberties(new_state.board, i, j)
            suicide = played_libs == 0

            # check if new state is repeats board history
            repeats_board = (jnp.sign(state.prev_boards[1]) == jnp.sign(new_state.board)).all()

            return ~(repeats_board | suicide)


        # not allowed to put a stone in a non-free position
        free_mask = jnp.full_like(board, True)
        free_mask = free_mask & (board == 0)

        # check if playing each position is (1) suicide, or (2) repeats board configuration
        coords = jnp.stack(jnp.indices((n, n)), axis=-1).reshape(-1, 2)
        play_mask = jax.vmap(_played_check)(coords).reshape((n, n))

        return free_mask & play_mask
