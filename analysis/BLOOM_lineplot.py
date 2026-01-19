#generates :
# - Figure 4

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.use("pgf")
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

columnwidth_in = 3.03125  
aspect_ratio = 0.65       
figsize = (columnwidth_in, columnwidth_in * aspect_ratio)
fontsize = 9.7

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "pgf.preamble": r"""
        \PassOptionsToPackage{table,x11names}{xcolor}
        \usepackage{xcolor}
        \usepackage{mathptmx}
        \definecolor{ibmyellow}{HTML}{FFB000} 
        \definecolor{ibmorange}{HTML}{FF8300}
        \definecolor{ibmpurple}{HTML}{785EF0}
        \definecolor{ibmblue}{HTML}{648FFF}
        \definecolor{ibmred}{HTML}{DC267F}
        \newcommand{\U}{\textcolor{ibmorange}{U}}
        \newcommand{\Pp}{\textcolor{ibmpurple}{P}}
        \newcommand{\D}{\textcolor{ibmred}{D}}
    """,
    "figure.figsize": figsize, 
    "axes.labelsize": fontsize-2,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize-2,
    "ytick.labelsize": fontsize-2,
    "legend.fontsize": fontsize,
    "legend.title_fontsize": fontsize
})


IN_PATH = "data/BLOOM_processed_data/paragraph_metrics.csv"
OUT_PATH = "latex/imgs/pics/BLOOM_20_paragraphs.pdf"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

cond_map = {
    "T": r"\U",
    "P": r"\Pp",
    "D": r"\D",
    "DP": r"\Pp+\D"
}

color_map = {
    r"\U": "#FF8300",
    r"\Pp": "#785EF0",
    r"\D": "#E62325",
    r"\Pp+\D": "#648FFF"
}

cond_order = ["T", "P", "D", "DP"]


print(f"[INFO] Loading data from: {IN_PATH}")
df = pd.read_csv(IN_PATH)
df = df[~df["lang"].isin(["khm", "fil", "nep"])]

# Only consider stories with 20 paragraphs
story_lengths = df.groupby("story_id")["paragraph"].nunique()
valid_ids = story_lengths[story_lengths == 20].index
df = df[df["story_id"].isin(valid_ids)]


records = []
for _, row in df.iterrows():
    records.append({
        "paragraph": row["paragraph"],
        "condition": row["cond"],
        "value": row["UIDv"]
    })

df_long = pd.DataFrame.from_records(records)
df_long = df_long[df_long["condition"].isin(cond_order)]


summary = (
    df_long.groupby(["paragraph", "condition"])
    ["value"].agg(["mean", "std", "count"]).reset_index()
)
summary["sem"] = summary["std"] / summary["count"]**0.5
summary["ci95"] = 1.96 * summary["sem"]


fig, ax = plt.subplots(figsize=figsize)  

for cond in cond_order:
    label = cond_map[cond]
    color = color_map[label]
    sub = summary[summary["condition"] == cond]
    x = sub["paragraph"]
    y = sub["mean"]
    yerr = sub["ci95"]

    ax.plot(x, y, label=label, color=color, linewidth=0.5)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

ax.set_xlabel("Paragraph Index", fontsize=fontsize)
ax.set_ylabel("UID$_{v}$", fontsize=fontsize)
ax.set_title("UID$_{v}$ Across Paragraphs (All Languages)", fontsize=fontsize)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
ax.grid(True, linestyle="--", linewidth=0.45)
leg = ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, 
    frameon=True, edgecolor="black", facecolor="white",
    handlelength=0.8, handletextpad=0.2, columnspacing=0.6,
    fontsize=fontsize-2, title_fontsize=fontsize, title="Conditions"
)

#leg.get_frame().set_facecolor('none')  # makes background transparent
#leg.get_frame().set_edgecolor('none')  # removes border

fig.subplots_adjust(bottom=0.26)
fig.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.02)
plt.close()
print(f"[✓] Plot saved to: {OUT_PATH}")