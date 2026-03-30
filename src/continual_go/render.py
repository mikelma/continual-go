import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


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
