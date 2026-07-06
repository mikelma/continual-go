import tyro
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec


def main(data: str = "./data_skill_level.csv", font_size: int = 14):
    df = pd.read_csv(data)

    # alphas = df["dirichlet_alpha"].unique()
    alphas = [0]

    #############################
    alphas = alphas[:2]

    #############################

    plt.rcParams.update({"font.size": font_size})

    fig = plt.figure()
    # gs = GridSpec(2, len(alphas))
    gs = GridSpec(1, len(alphas))

    for i, alpha in enumerate(sorted(alphas)):
        # sel = df.loc[df["dirichlet_alpha"] == alpha]
        sel = df

        # ax1 = fig.add_subplot(gs[0, i])
        ax1 = fig.add_subplot(gs[i])
        df["return_A"] = df["return_A"] - df["return_B"]
        # ax1.set_title(f"Dirichlet alpha: {alpha}")
        sns.boxplot(data=sel, x="skill_level", y="return_A", ax=ax1)
        # sns.lineplot(data=sel, x="skill_level", y="return_A", ax=ax1)

        # ax2 = fig.add_subplot(gs[1, i])
        # sns.boxplot(data=sel, x="skill_level", y="return_B", ax=ax2)

        # for ax in [ax1, ax2]:
        for ax in [ax1]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Skill level")
            ax.tick_params(axis="x", labelrotation=45)

        ax1.set_ylabel("Points of AZ opponent")
        # ax2.set_ylabel("Points of agent")

    plt.show()


if __name__ == "__main__":
    tyro.cli(main)
