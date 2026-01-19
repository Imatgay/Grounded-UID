# generates:
# - Figure 3

import os
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf

matplotlib.use("pgf")
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

fontsize = 22
matplotlib.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "pgf.preamble": r"""
        \PassOptionsToPackage{table,x11names}{xcolor}
        \usepackage{xcolor}
        \usepackage{mathptmx}
        \definecolor{ibmyellow}{HTML}{FFB000}
        \definecolor{ibmorange}{HTML}{FE6100}
        \definecolor{ibmpurple}{HTML}{785EF0}
        \definecolor{ibmblue}{HTML}{648FFF}
        \definecolor{ibmred}{HTML}{DC267F}
        \newcommand{\U}{\textcolor{ibmorange}{U}}
        \newcommand{\Pp}{\textcolor{ibmpurple}{P}}
        \newcommand{\D}{\textcolor{ibmred}{D}}
    """,
    "figure.figsize": (10, 5.5),
    "figure.constrained_layout.use": True,
    "legend.frameon": False,
    "legend.loc": "lower center"
})

def main():
    csv_input_path = "data/BLOOM_processed_data/paragraph_metrics.csv"
    df = pd.read_csv(csv_input_path)

    df = df[~df["lang"].isin(["khm", "fil", "nep"])]

    cond_order = ["T", "P", "D", "DP"]
    cond_map = {
        "T": "surprisal_ablation_txt",
        "P": "surprisal_ablation_mm",
        "D": "surprisal_total_txt",
        "DP": "surprisal_total_mm"
    }
    label_map = {
        "T": r"\U\ (no context)",
        "P": r"\Pp\ (no context)",
        "D": r"\D",
        "DP": r"\Pp+\D"
    }
    palette_uidv_custom = {
        r"\U\ (no context)": "#FF8300",
        r"\Pp\ (no context)": "#785EF0",
        r"\D": "#DC267F",
        r"\Pp+\D": "#648FFF"
    }

    df_uidv = df[["lang", "story_id", "paragraph", "cond", "UIDv"]].copy()
    df_uidv["condition"] = df_uidv["cond"].map(cond_map)
    df_uidv["condition_pretty"] = df_uidv["cond"].map(label_map)
    df_uidv.rename(columns={"lang": "language", "UIDv": "uidv"}, inplace=True)
    df_uidv["language"] = pd.Categorical(df_uidv["language"],
                                        categories=sorted(df_uidv["language"].unique()),
                                        ordered=True)

    global_means_v = df_uidv.groupby("condition")["uidv"].mean().to_dict()

    fig, ax = plt.subplots(figsize=(1.0 * df_uidv["language"].nunique(), 6.0))
    sns.boxplot(
        data=df_uidv,
        x="language",
        y="uidv",
        hue="condition_pretty",
        hue_order=[label_map[c] for c in cond_order],
        palette=palette_uidv_custom,
        showfliers=False,
        width=0.80,
        ax=ax
    )
    
    for cond in cond_order:
        if cond_map[cond] in global_means_v:
            ax.axhline(
                y=global_means_v[cond_map[cond]],
                linestyle="--",
                linewidth=1.5,
                color=palette_uidv_custom[label_map[cond]],
                alpha=0.8
            )

    ax.set_title(r"{UID}$_{v}$ by Language and Condition", fontsize=fontsize)
    ax.set_xlabel("Language", fontsize=fontsize-2)
    ax.set_ylabel(r"{UID}$_{v}$", fontsize=fontsize-2)
    ax.tick_params(axis='x', labelrotation=45, labelsize=fontsize-2)
    ax.tick_params(axis='y', labelsize=fontsize-2)

    handles, labels = ax.get_legend_handles_labels()
    box_handles = [h for h in handles if not isinstance(h, matplotlib.lines.Line2D)]
    box_labels = [l for h, l in zip(handles, labels) if not isinstance(h, matplotlib.lines.Line2D)]

    ax.legend(
        box_handles, box_labels,
        title="Conditions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(box_handles),
        frameon=True,
        edgecolor="black",
        facecolor="white",
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
        fontsize=fontsize-2,
        title_fontsize=fontsize
    )

    fig.subplots_adjust(bottom=0.4)
    os.makedirs("latex/imgs/pics/", exist_ok=True)
    fig.savefig("latex/imgs/pics/BLOOM_boxplot.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

if __name__ == "__main__":
    main()