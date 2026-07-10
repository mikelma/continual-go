import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from tyro.extras import SubcommandApp
import scipy as sp
from typing import Literal


app = SubcommandApp()


@app.command
def dirichlet(
    data: str = "./data_skill_level.csv",
    font_size: int = 14,
    plot: Literal["line", "box"] = "box",
):
    df = pd.read_csv(data)

    alphas = df["dirichlet_alpha"].unique()

    plt.rcParams.update({"font.size": font_size})

    fig = plt.figure()
    gs = GridSpec(2, len(alphas))

    df["return_A"] = df["return_A"] - df["return_B"]
    for i, alpha in enumerate(sorted(alphas)):
        sel = df.loc[df["dirichlet_alpha"] == alpha]
        # sel = df

        ax1 = fig.add_subplot(gs[0, i])
        ax1.set_title(f"Dirichlet alpha: {alpha}")
        if plot == "box":
            sns.boxplot(data=sel, x="skill_level", y="return_A", ax=ax1)
        elif plot == "line":
            sns.lineplot(data=sel, x="skill_level", y="return_A", ax=ax1)

        ax2 = fig.add_subplot(gs[1, i])
        sns.boxplot(data=sel, x="skill_level", y="return_B", ax=ax2)

        for ax in [ax1, ax2]:
            # for ax in [ax1]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Skill level")
            ax.tick_params(axis="x", labelrotation=45)

        ax1.set_ylabel("Points of AZ opponent")
        ax2.set_ylabel("Points of agent")

    plt.show()


@app.command
def ranking(
    data: str = "./data_skill_level.csv",
    font_size: int = 17,
    plot: Literal["line", "box"] = "box",
):
    df = pd.read_csv(data)

    plt.rcParams.update({"font.size": font_size})

    df["return_A"] = df["return_A"] - df["return_B"]

    r, p = sp.stats.pearsonr(df["return_A"], df["skill_level"])

    # plt.title("Ranking-based skill control")

    if plot == "box":
        sns.boxplot(data=df, x="skill_level", y="return_A")
    elif plot == "line":
        sns.lineplot(data=df, x="skill_level", y="return_A", errorbar="sd")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Skill level")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("Opponent's return")
    plt.text(0.8, 0.8, "Pearson's r ={:.2f}".format(r), transform=ax.transAxes)

    plt.show()


@app.command
def many(
    data: str = "./data_skill_level.csv",
    font_size: int = 17,
    plot: Literal["line", "box"] = "box",
):
    df = pd.read_csv(data)

    plt.rcParams.update({"font.size": font_size})

    df["return_A"] = df["return_A"] - df["return_B"]

    corrs = []
    methods = df["sampling_method"].unique()
    for method in methods:
        sel = df[df["sampling_method"] == method]
        r, p = sp.stats.pearsonr(sel["skill_level"], sel["return_A"])
        corrs.append(r)
        print(method)
        if "ranking" in method:
            df.loc[df["sampling_method"] == method, "skill_level"] /= df[
                df["sampling_method"] == method
            ]["skill_level"].max()

    # plt.title("Ranking-based skill control")

    if plot == "box":
        sns.boxplot(data=df, x="skill_level", y="return_A", hue="sampling_method")
    elif plot == "line":
        sns.lineplot(
            data=df, x="skill_level", y="return_A", errorbar="sd", hue="sampling_method"
        )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Skill level")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("Opponent's return")
    rstr = "Pearson's r:\n" + "\n".join(
        [f"{m}: " + "{:.2f}".format(r) for m, r in zip(methods, corrs)]
    )
    plt.text(0.8, 0.8, rstr, transform=ax.transAxes)

    plt.show()


if __name__ == "__main__":
    app.cli()
