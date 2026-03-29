import jax
import jax.numpy as jnp
from flax import struct
from flax.struct import PyTreeNode
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray, Bool
from typing import TypeAlias
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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

        return state.replace(
            board=new_board,
            turn=state.turn * -1,
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


def plot_board(board, ax=None, show=True):
    n = board.shape[0]

    if ax is None:
        fig_size = max(5, n * 0.4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    else:
        fig = ax.figure

    # Board background
    ax.set_facecolor("#DDB76F")

    # Grid
    for i in range(n):
        ax.plot([0, n - 1], [i, i], color="black", linewidth=1, zorder=1)
        ax.plot([i, i], [0, n - 1], color="black", linewidth=1, zorder=1)

    # Stones
    stone_radius = 0.46
    font_size = max(8, int(180 / max(n, 5)))

    for r in range(n):
        for c in range(n):
            val = board[r, c]
            if val == 0:
                continue

            x = c
            y = n - 1 - r  # row 0 at top

            is_black = val < 0
            face = "black" if is_black else "white"
            text_color = "white" if is_black else "black"

            stone = Circle(
                (x, y),
                stone_radius,
                facecolor=face,
                edgecolor="black",
                linewidth=1.2,
                fill=True,
                alpha=1.0,
                zorder=3,
            )
            ax.add_patch(stone)

            ax.text(
                x,
                y,
                str(abs(int(val))),
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
                fontweight="bold",
                zorder=4,
            )

    # Formatting
    margin = 0.6
    ax.set_xlim(-margin, n - 1 + margin)
    ax.set_ylim(-margin, n - 1 + margin)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(np.arange(n)[::-1])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show:
        plt.show()

    return fig, ax
