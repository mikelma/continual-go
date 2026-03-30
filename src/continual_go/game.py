import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray, Bool
from typing import TypeAlias


IntLike: TypeAlias = Integer[ScalarLike, ""]


@struct.dataclass
class State:
    num_actions: IntLike
    size: IntLike
    board: Integer[Array, "size size"]
    turn: IntLike  # -1 (black) or +1 (white)
    k: IntLike  # max number of stones per player


# def _adj_ixs(ij, size):
#     delta = jnp.int32([
#         [0, -1],  # left
#         [0, +1],  # right
#         [+1, 0],  # up
#         [-1, 0],  # down
#     ])
#     adj = ij + delta
#     on_board = (adj >= 0) & (adj < size)

#     return jnp.where(on_board, adj, -1), on_board.all(1)


def _adjacent4(mask: jax.Array) -> jax.Array:
    """For each cell, whether it is adjacent (4-neighborhood) to any True cell."""
    from_up = jnp.pad(mask[:-1, :], ((1, 0), (0, 0)))
    from_down = jnp.pad(mask[1:, :], ((0, 1), (0, 0)))
    from_left = jnp.pad(mask[:, :-1], ((0, 0), (1, 0)))
    from_right = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)))
    return from_up | from_down | from_left | from_right


class ContinualGo:
    def init(self, size: IntLike = 9) -> State:
        board = jnp.zeros((size, size), dtype=int)
        return State(
            num_actions=(size * size) - 1, size=size, board=board, turn=-1, k=16
        )

    def step(self, state: State, action: IntLike) -> tuple[State, ScalarLike]:
        n = state.size
        action = jnp.minimum(jnp.array((n * n - 1)), action)  # no pass allowed

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
        coords = jnp.stack(jnp.indices((state.size, state.size)), axis=-1).reshape(
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
            (jnp.sign(new_board) == opponent).sum() > (state.k - 1),
            lambda: remove_stone(opponent),
            lambda: new_board,
        )

        return state.replace(
            board=new_board,
            turn=opponent,
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

        # # board = jax.lax.cond()
        # def _count(i, j, checked):
        #     checked = checked.at[i, j].set(True)
        #     adj_idx, on_board = _adj_ixs(jnp.asarray((i, j)), board.shape[0])

        #     print("adj idx:", adj_idx)
        #     adj_stones = board[adj_idx[:, 0], adj_idx[:, 1]]
        #     print(adj_stones)

        #     # number of "immediate" liberties for this stone
        #     liberties = ((adj_stones == 0) & on_board).sum()
        #     print("immediate liberties:", liberties)

        #     # mask stones that are adjacent and owned by the current playerx
        #     players_mask = jax.lax.cond(
        #         board[i, j] > 0,
        #         lambda: adj_stones > 0,  # player +1
        #         lambda: adj_stones < 0   # player -1
        #     )
        #     # the stone should be player's and not been already checked
        #     resolve_mask = players_mask & ~checked[adj_idx[:, 0], adj_idx[:, 1]]
        #     print("resolve mask:", resolve_mask)

        #     def _count_neighbor(flag, ij):
        #         return jax.lax.cond(
        #             flag,
        #             lambda: _count(ij[0], ij[1], checked),
        #             lambda: (0, checked),
        #         )

        #     def _scan_count(checked, neighbor_idx):
        #         flag = resolve_mask[neighbor_idx]
        #         ij = adj_idx[neighbor_idx]
        #         libs, checked = _count_neighbor(flag, ij)
        #         return checked, libs

        #     # iteratiboki jun flag eta adj_idx-tik, gehitzen libertiak eta azken deiaren checked erabiltzen
        #     checked, add_libs = jax.lax.scan(_scan_count, checked, jnp.arange(4))
        #     liberties += add_libs.sum()
        #     print("liberties:", liberties)

        #     quit(13)
        #     return liberties, checked

        # _count(i, j, jnp.full_like(board, False))
        # # n = board.shape[0]
        # # return jax.lax.cond(
        # #     board[i, j] == 0,
        # #     lambda: 0,
        # #     lambda: _count(i, j, jnp.full_like(board, False))
        # # )

    def legal_actions(self, state: State) -> Bool[Array, "size size"]:
        player = state.turn
        board = state.board
        # not allowed to put a stone in a non-free position
        mask = jnp.full_like(board, True)
        mask = mask & (board == 0)
        return mask
