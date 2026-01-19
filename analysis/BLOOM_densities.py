# generates:
# - figures 8 to 20 (Appendix)

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pgf")
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

import seaborn as sns
sns.set_style("whitegrid")

plt.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "pgf.preamble": r"""
        \PassOptionsToPackage{table,x11names}{xcolor}
        \usepackage{xcolor}
        \usepackage{newtxtext,newtxmath}
        \definecolor{ibmyellow}{HTML}{FFB000} 
        \definecolor{ibmorange}{HTML}{FF8300}
        \definecolor{ibmpurple}{HTML}{785EF0}
        \definecolor{ibmblue}{HTML}{648FFF}
        \definecolor{ibmred}{HTML}{DC267F}
        \newcommand{\U}{\textcolor{ibmorange}{U}}
        \newcommand{\Pp}{\textcolor{ibmpurple}{P}}
        \newcommand{\D}{\textcolor{ibmred}{D}}
    """,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

WORD_DIR = Path("data/BLOOM_word_surprisal")
ABS_LATEX_DIR  = Path("latex/imgs/bloom_densities")
OUTPUT_DIR     = Path("data/_other_data")


ABS_LATEX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PARAS = 5
MIN_WORDS_PARA = 10
SIGMA_MULTIPLIER = 0.05
SIGMA_MULTIPLIER_PARA = 0.05
EPSILON = 1e-6

COLOR_MAP = {
    "$\\delta_P$": "#377eb8",
    "$\\delta_D$": "#ff7f00",
    "$\\delta_{DP}$": "#4daf4a"
}
CTX_LABELS = ["$\\delta_P$", "$\\delta_D$", "$\\delta_{DP}$"]


def absolute_reduction(nom, denom):
    delta = nom - denom
    return np.where(delta > 0, delta, 0.0)

def optimal_bins(x, rule="fd"):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 10
    if rule == "fd":
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        bin_width = 2 * iqr / np.cbrt(n)
    elif rule == "scott":
        std = np.std(x)
        bin_width = 3.5 * std / np.cbrt(n)
    elif rule == "sturges":
        return int(np.ceil(np.log2(n) + 1))
    else:
        raise ValueError(f"Unknown rule: {rule}")
    if bin_width <= 0:
        return 10
    return max(int(np.ceil((x.max() - x.min()) / bin_width)), 10)

def pos_density(x_norm, delta, nbins, sigma):
    bins = np.linspace(0.0, 1.0, nbins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(x_norm, bins, right=False) - 1
    hist = np.zeros(nbins, dtype=float)
    np.add.at(hist, idx.clip(0, nbins - 1), delta)
    if hist.sum() == 0:
        return centres, hist
    density = hist / hist.sum()
    smooth = gaussian_filter1d(density, sigma=sigma, mode="reflect")
    smooth = np.clip(smooth, 0.0, None)
    smooth /= smooth.sum()
    return centres, smooth

def make_combined_plot(sub, lang, metric_map, out_dir, suffix):
    nbins_sent = optimal_bins(sub["sent_pos_norm"].values, rule="fd")
    nbins_para = optimal_bins(sub["para_pos_norm"].values, rule="fd")
    
    sigma_sent = SIGMA_MULTIPLIER * nbins_sent
    sigma_para = SIGMA_MULTIPLIER_PARA * nbins_para

    figsize = (3.03125 * 2, 3.03125 * 0.9)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    handles = []  # <<< collect legend handles here

    for label in CTX_LABELS:
        v_sent = sub[metric_map[label]["col"]].values
        v_para = sub[metric_map[label]["col"]].values
        x_sent = sub["sent_pos_norm"].values
        x_para = sub["para_pos_norm"].values

        x_s, d_s = pos_density(x_sent, v_sent, nbins=nbins_sent, sigma=sigma_sent)
        x_p, d_p = pos_density(x_para, v_para, nbins=nbins_para, sigma=sigma_para)

        line, = axs[0].plot(x_s, d_s, label=label, color=COLOR_MAP[label], linewidth=1.5)
        axs[1].plot(x_p, d_p, label=label, color=COLOR_MAP[label], linewidth=1.5)
        handles.append(line)

        raw_sum_sent = np.sum(np.where(v_sent > 0, v_sent, 0.0))
        raw_sum_para = np.sum(np.where(v_para > 0, v_para, 0.0))
        area_sent = np.trapz(d_s, x_s)
        area_para = np.trapz(d_p, x_p)

    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(bottom=0)
    axs[0].set_xlabel(metric_map[label]["x_sent"])
    axs[0].set_ylabel(metric_map[label]["y"])
    axs[0].set_title(f"{lang} – sentence level")
    axs[0].axhline(0, color="grey", ls="--", lw=0.8)
    sns.despine(ax=axs[0])

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlabel(metric_map[label]["x_para"])
    axs[1].set_title(f"{lang} – paragraph level")
    axs[1].axhline(0, color="grey", ls="--", lw=0.8)
    sns.despine(ax=axs[1])

    fig.legend(
        handles=handles,
        labels=CTX_LABELS,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        title="Context"
    )

    fig.tight_layout(rect=[0, 0.15, 1, 1])  # Adjust for legend space
    fig.savefig(out_dir / f"{lang}_{suffix}_density.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def main():
    dfs = []
    for fpath in tqdm(sorted(WORD_DIR.glob("*.csv")), desc="Loading files"):
        parts = fpath.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        lang = parts[1]
        tmp = pd.read_csv(fpath)
        tmp["lang"] = lang
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    df["word_index"] = df.groupby(["story_id", "paragraph"]).cumcount()

    bad = set()
    for sid, g in df.groupby("story_id"):
        paras = g["paragraph"].unique()
        if not np.array_equal(np.sort(paras), np.arange(len(paras))):
            bad.add(sid)
            continue
        if len(paras) < MIN_PARAS:
            bad.add(sid)
            continue
        if (g.groupby("paragraph").size() < MIN_WORDS_PARA).any():
            bad.add(sid)
            continue
    df = df.loc[~df["story_id"].isin(bad)].copy()

    df["word_in_sent"] = df.groupby(["story_id", "paragraph", "sentence_idx"]).cumcount()
    df["sent_len"] = df.groupby(["story_id", "paragraph", "sentence_idx"])["word_in_sent"].transform("max") + 1
    df["sent_pos_norm"] = df["word_in_sent"] / df["sent_len"]

    df["word_in_para"] = df.groupby(["story_id", "paragraph"]).cumcount()
    df["para_len"] = df.groupby(["story_id", "paragraph"])["word_in_para"].transform("max") + 1
    df["para_pos_norm"] = df["word_in_para"] / df["para_len"]

    metrics = {
       "absolute": {
           "$\\delta_P$": {
               "col": "abs_P",
               "x_sent": r"\textrm{normalised sentence position}",
               "x_para": r"\textrm{normalised paragraph position}",
               "y": r"\textrm{density}"
           },
           "$\\delta_D$": {
               "col": "abs_D",
               "x_sent": r"\textrm{normalised sentence position}",
               "x_para": r"\textrm{normalised paragraph position}",
               "y": r"\textrm{density}"
           },
           "$\\delta_{DP}$": {
               "col": "abs_DP",
               "x_sent": r"\textrm{normalised sentence position}",
               "x_para": r"\textrm{normalised paragraph position}",
               "y": r"\textrm{density}"
           }
       }
    }


    df["abs_P"] = absolute_reduction(df["surprisal_ablation_txt"], df["surprisal_ablation_mm"])
    df["abs_D"] = absolute_reduction(df["surprisal_ablation_txt"], df["surprisal_total_txt"])
    df["abs_DP"] = absolute_reduction(df["surprisal_total_txt"], df["surprisal_total_mm"])

    for lang in sorted(df["lang"].unique()):
        sub = df[df["lang"] == lang]
        make_combined_plot(sub, lang, metrics["absolute"], ABS_LATEX_DIR, "absolute_reduction")

if __name__ == "__main__":
    main()
