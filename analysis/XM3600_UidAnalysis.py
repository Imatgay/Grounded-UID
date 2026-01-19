# generates:
# - Figure 1
# - Figure 2
# - Table 3 (Appendix)


# cache data for XM3600 in data/XCROSS_processed_data/XCROSS_uid_data.parquet

import numpy as np
import pandas as pd
import polars as pl

from scipy.stats import wilcoxon, gaussian_kde, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

import seaborn as sns
from adjustText import adjust_text

import os
import re
from typing import List, Tuple
from tqdm import tqdm

from utils import iso_639_3

import matplotlib
matplotlib.use("pgf")  


fontsize = 7.5

matplotlib.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "legend.title_fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "pgf.preamble": r"""
        \PassOptionsToPackage{table,x11names}{xcolor}
        \usepackage{xcolor}
        \usepackage{mathptmx}
    """
})

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.ticker import MultipleLocator
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)


########
######## PLOT AND LATEX
########


def save_summary(stats):
    df_stats = pl.DataFrame(stats).sort("lang")
    with open("latex/tabs/XCROSS_datadist.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|rrrrrr}\n")
        f.write("\\hline\n")
        f.write("Lang & Captions & Tokens & Mean & Std & Min & Max \\\\\n")
        f.write("\\hline\n")
        for row in df_stats.iter_rows():
            f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]:.1f} & {row[4]:.1f} & {row[5]} & {row[6]} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("""\\caption{Summary statistics of caption lengths across languages in the \textsc{Ground-XM3600} dataset. 
            \textit{Caps}: number of captions; \textit{Words}: total word count; \textit{Mean} and 
            \textit{Std}: average and standard deviation of caption lengths (in words).}\n""")
        f.write("\\label{tab:xcross_summary}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX dataset stats to latex/tabs/XCROSS_datadist.tex")

def plot_uid_scatter_from_table(uid_table_path: str, output_pdf: str):
    with open(uid_table_path, 'r') as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        if line.strip().startswith('\\') or line.strip() == "":
            continue
        if '&' not in line:
            continue
        cells = [re.sub(r'\\textbf{(.*?)}', r'\1', c.strip()) for c in line.split('&')]

        if len(cells) >= 7: 
            try:
                lang = cells[0]
                uidv_lm = float(cells[1])
                uidv_cap = float(cells[2])
                uidlv_lm = float(cells[3])
                uidlv_cap = float(cells[4])
                rows.append({
                    "lang": lang,
                    "UID_v_LM": uidv_lm,
                    "UID_v_CAP": uidv_cap,
                    "UID_lv_LM": uidlv_lm,
                    "UID_lv_CAP": uidlv_cap
                })
            except ValueError:
                continue

    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        raise ValueError("Failed to parse any data from the LaTeX table.")

    languages = df_plot["lang"].tolist()
    palette = sns.color_palette("husl", len(languages))
    lang2color = dict(zip(languages, palette))

    all_values = df_plot[
        ["UID_v_LM", "UID_v_CAP", "UID_lv_LM", "UID_lv_CAP"]
    ].values.flatten()
    min_val = all_values.min()
    max_val = all_values.max()
    pad = 0.05 * (max_val - min_val)
    axis_min = min_val - pad
    axis_max = max_val + pad

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), dpi=150)

    for i, metric in enumerate(["UID_v", "UID_lv"]):
        ax = axes[i]
        x = df_plot[f"{metric}_LM"]
        y = df_plot[f"{metric}_CAP"]
        langs = df_plot["lang"]

        offset = 0.01 * (axis_max - axis_min)

        for xi, yi, lang in zip(x, y, langs):
            col = lang2color[lang]
            ax.scatter(xi, yi, color=col, s=10, edgecolor='black', linewidths=0.3, zorder=3)

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        ax.plot([axis_min, axis_max], [axis_min, axis_max],
                linestyle="--", color="gray", linewidth=0.8, zorder=1)

        latex_metric = rf"$\mathrm{{UID}}_{{lv}}$" if metric == "UID_lv" else rf"$\mathrm{{UID}}_{{v}}$"
        latex_wild_metric = rf"$\mathrm{{UID}}_{{*}}$"
        xlabel = rf"{latex_metric}$\bigl(\mathrm{{\textcolor[HTML]{{FE6100}}{{U}}}}\bigr)$"
        ylabel = rf"{latex_metric}$\bigl(\mathrm{{\textcolor[HTML]{{785EF0}}{{P}}}}\bigr)$"
        
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if i == 0:  # Only left plot gets the y-axis label
            ax.set_ylabel(ylabel, fontsize=fontsize)
        else:
            ax.set_ylabel("")
            
        if metric == "UID_lv":
            ax.set_title(rf"Local Uniformity")
        else:
            ax.set_title(rf"Global Uniformity")

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.02)
    plt.clf()
    plt.cla()
    plt.close('all')
    print(f"UID scatter plots saved to {output_pdf}")

def plot_global_uid_density(all_data: pd.DataFrame, clip_percentile: float = 99.0):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from scipy.stats import gaussian_kde
    import numpy as np
    import os


    fontsize = 9.7
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": fontsize,
        "axes.labelsize": fontsize-2,
        "axes.titlesize": fontsize,
        "legend.fontsize": fontsize,
        "legend.title_fontsize": fontsize,
        "xtick.labelsize": fontsize-2,
        "ytick.labelsize": fontsize-2,
        "pgf.preamble": r"""
            \PassOptionsToPackage{table,x11names}{xcolor}
            \usepackage{xcolor}
            \usepackage{mathptmx}
        """
    })

    columnwidth_in = 3.03125  
    aspect_ratio = 0.65       
    figsize = (columnwidth_in, columnwidth_in * aspect_ratio)

    os.makedirs("latex/imgs/pics/", exist_ok=True)
    plot_df = all_data.copy()
    threshold = plot_df["uid"].quantile(clip_percentile / 100)
    plot_df = plot_df[plot_df["uid"] <= threshold]
    
    fig, ax = plt.subplots(figsize=figsize)

    color_map = {"LM": "#FE6100", "CAP": "#785EF0"}
    linestyle_map = {"UID_v": (0, (2, 1.0)), "UID_lv": "solid"}

    for cond in plot_df["cond"].unique():
        for metric in plot_df["metric"].unique():
            subset = plot_df[(plot_df["cond"] == cond) & (plot_df["metric"] == metric)]
            if len(subset) < 2:
                continue
            uid_vals = subset["uid"].values
            try:
                kde = gaussian_kde(uid_vals)
            except np.linalg.LinAlgError:
                continue
            x_grid = np.linspace(uid_vals.min(), uid_vals.max(), 200)
            y_vals = kde(x_grid)
            ax.plot(x_grid, y_vals,
                    color=color_map[cond],
                    linestyle=linestyle_map[metric],
                    linewidth=1.5)

    ax.set_title("Global UID")
    ax.set_xlabel("UID")
    ax.set_ylabel("Density")
    ax.grid(True, which="major", linestyle="--", linewidth=0.45)

    def make_label(metric: str, cond: str) -> str:
        letter = "U" if cond == "LM" else "P"
        color = "FE6100" if cond == "LM" else "785EF0"
        if metric == "UID_v":
            return rf"$\mathrm{{UID}}_v$ (\textcolor[HTML]{{{color}}}{{{letter}}})"
        elif metric == "UID_lv":
            return rf"$\mathrm{{UID}}_{{lv}}$ (\textcolor[HTML]{{{color}}}{{{letter}}})"
        else:
            return rf"${metric}$ (\textcolor[HTML]{{{color}}}{{{letter}}})"

    handles = []
    labels = []
    for metric in ["UID_v", "UID_lv"]:
        for cond in ["LM", "CAP"]:
            handles.append(mlines.Line2D([], [], color=color_map[cond],
                                         linestyle=linestyle_map[metric], linewidth=1.5))
            labels.append(make_label(metric, cond))

    
    fig.legend(
        handles=handles,
        labels=labels,
        title="Metric (Condition)",
        loc="lower right",                                 
        bbox_to_anchor=(0.92, -0.15),                        
        bbox_transform=fig.transFigure,
        ncol=4,
        frameon=True, 
        edgecolor="black", 
        facecolor="white",
        handlelength=0.8,
        handletextpad=0.2,
        columnspacing=0.3,
        fontsize=fontsize - 2,
        title_fontsize=fontsize
    )

    fig.subplots_adjust(bottom=0.26)  
    fig.savefig("latex/imgs/pics/XCROSS_uid_distribution.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print("Density plot saved to: latex/imgs/pics/XCROSS_uid_distribution.pdf")

##########
########## COMPUTE
##########

def uid_sentence(log_probs: np.ndarray, local: bool = False) -> float:
    if local is True:
        diffs = np.diff(log_probs)
        return np.mean(diffs ** 2)
    mu = log_probs.mean()
    return np.mean((log_probs - mu) ** 2)

def ci_bootstrap(arr1: List[float], arr2: List[float], n_boot: int = 2000) -> Tuple[float, float]:
    diffs = np.array(arr2) - np.array(arr1)
    boot = [np.mean(resample(diffs)) for _ in range(n_boot)]
    return np.percentile(boot, [2.5, 97.5])

def cohens_d_paired(x: List[float], y: List[float]) -> float:
    diff = np.array(y) - np.array(x)
    n = len(diff)
    if n < 2: 
        return 0.0
    # Standard deviation of the difference (ddof=1 for sample std)
    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        return 0.0
    return np.mean(diff) / sd_diff

##########
########## MAIN
##########

def main(csv_path: str, dataset: str, outfile_tex: str, min_len: int = 3, do_summary: bool = False):
    os.makedirs('latex/tabs', exist_ok=True)

    df = (pl.read_csv(csv_path)
            .filter(pl.col("dataset") == dataset)
            .with_columns(pl.col("log_lm").cast(pl.Float64),
                          pl.col("log_cap").cast(pl.Float64)))

    languages = df["lang"].unique().to_list()
    rows = []
    summary_stats = []
    all_sentence_uids = []


    for lang in tqdm(languages, desc="UID computation in GroundXCROSS. Processing languages ..."):
        lang_df = df.filter(pl.col("lang") == lang)

        # extract captions ("sentences" in code) by finding idx 0 (first word).
        idxs = (lang_df.select(pl.arg_where(pl.col("sentence_idx") == 0)) 
                        .to_series().to_list() + [len(lang_df)])
        sentences = [lang_df[idxs[i]:idxs[i+1]]
                     for i in range(len(idxs)-1)
                     if idxs[i+1] - idxs[i] >= min_len]

        uidv_lm, uidv_cap, uidlv_lm, uidlv_cap = [], [], [], []

        for sent_idx, s in enumerate(sentences):
            lm = np.array(s["log_lm"])
            cap = np.array(s["log_cap"])
            words = s["word"].to_list()
            text = " ".join(w if w is not None else "[MISSING]" for w in words)

            v_lm  = uid_sentence(lm)
            v_cap = uid_sentence(cap)
            lv_lm = uid_sentence(lm, local=True)
            lv_cap= uid_sentence(cap, local=True)

            uidv_lm.append(v_lm)
            uidv_cap.append(v_cap)
            uidlv_lm.append(lv_lm)
            uidlv_cap.append(lv_cap)

            sent_len = len(s["log_lm"])

            all_sentence_uids.append(dict(lang=lang, cond="LM",  metric="UID_v",  uid=v_lm,  sent_idx=sent_idx, text=text, length=sent_len))
            all_sentence_uids.append(dict(lang=lang, cond="CAP", metric="UID_v",  uid=v_cap, sent_idx=sent_idx, text=text, length=sent_len))
            all_sentence_uids.append(dict(lang=lang, cond="LM",  metric="UID_lv", uid=lv_lm, sent_idx=sent_idx, text=text, length=sent_len))
            all_sentence_uids.append(dict(lang=lang, cond="CAP", metric="UID_lv", uid=lv_cap, sent_idx=sent_idx, text=text, length=sent_len))

        ###
        stat_v,   p_v   = wilcoxon(uidv_lm,  uidv_cap)
        stat_lv,  p_lv  = wilcoxon(uidlv_lm, uidlv_cap)
        
        ###
        delta_v         = 100 * (np.mean(uidv_cap)  - np.mean(uidv_lm))  / np.mean(uidv_lm)
        delta_lv        = 100 * (np.mean(uidlv_cap) - np.mean(uidlv_lm)) / np.mean(uidlv_lm)
        
        ###
        ci_v            = ci_bootstrap(uidv_lm, uidv_cap)
        ci_lv           = ci_bootstrap(uidlv_lm, uidlv_cap)

        ###
        d_v = cohens_d_paired(uidv_lm, uidv_cap)
        d_lv = cohens_d_paired(uidlv_lm, uidlv_cap)

        lang = iso_639_3.get(lang, lang)
        rows.append(dict(
            lang=lang,
            uidv_lm=np.mean(uidv_lm), uidv_cap=np.mean(uidv_cap),
            uidlv_lm=np.mean(uidlv_lm), uidlv_cap=np.mean(uidlv_cap),
            delta_v=delta_v, delta_lv=delta_lv,
            d_v=d_v, d_lv=d_lv,  
            p_v=p_v, p_lv=p_lv,
            ci_v=f"[{ci_v[0]:.2f},{ci_v[1]:.2f}]",
            ci_lv=f"[{ci_lv[0]:.2f},{ci_lv[1]:.2f}]",
            n_sent=len(uidv_lm)
        ))

        if do_summary:
            lengths = [len(s) for s in sentences]
            summary_stats.append(dict(
            lang=lang,
            n_captions=len(lengths),
            n_tokens=sum(lengths),
            mean_len=np.mean(lengths),
            std_len=np.std(lengths, ddof=1),
            min_len=np.min(lengths),
            max_len=np.max(lengths),
        ))

    if do_summary:
        save_summary(summary_stats)

    df_res = pd.DataFrame(rows).set_index("lang")

    # FDR correction
    for metric in ["v", "lv"]:
        _, qvals, _, _ = multipletests(df_res[f"p_{metric}"], alpha=0.05, method="fdr_bh")
        df_res[f"q_{metric}"]   = qvals
        df_res[f"sig_{metric}"] = qvals < 0.05

    def sig_star(q):
        if q < 0.001:
            return '***'
        elif q < 0.01:
            return '**'
        elif q < 0.05:
            return '*'
        else:
            return 'n.s.'

    # star annotation
    df_res["sig_v_stars"]  = df_res["q_v"].apply(sig_star)
    df_res["sig_lv_stars"] = df_res["q_lv"].apply(sig_star)

    for col in ["uidv_lm", "uidv_cap", "uidlv_lm", "uidlv_cap"]:
        df_res[col] = df_res[col].apply(lambda x: f"{x:.2f}")

    def fmt_delta_with_star(val, star):
        val_fmt = f"{val:.2f}"
        if val > 0:
            val_fmt = f"\\textbf{{{val_fmt}}}"
        return f"{val_fmt}~{star}"

    df_res["delta_v_fmt"] = df_res.apply(lambda r: fmt_delta_with_star(r["delta_v"], r["sig_v_stars"]), axis=1)
    df_res["delta_lv_fmt"] = df_res.apply(lambda r: fmt_delta_with_star(r["delta_lv"], r["sig_lv_stars"]), axis=1)
    
    df_res["d_v_fmt"] = df_res["d_v"].apply(lambda x: f"{x:.2f}")
    df_res["d_lv_fmt"] = df_res["d_lv"].apply(lambda x: f"{x:.2f}")

    df_res = df_res.sort_index()

    out_fmt = df_res[[
        "uidv_lm", "uidv_cap", 
        "uidlv_lm", "uidlv_cap", 
        "delta_v_fmt", "d_v_fmt",    
        "delta_lv_fmt", "d_lv_fmt"   
    ]].copy()

    out_fmt.columns = [
        r"UID$_v$ (\U{})", r"UID$_v$ (\Pp{})",
        r"UID$_{lv}$ (\U{})", r"UID$_{lv}$ (\Pp{})",
        r"$\Delta_v$ (\%)", r"$d_v$",
        r"$\Delta_{lv}$ (\%)", r"$d_{lv}$"
    ]

    out_fmt.insert(0, "Lang", df_res.index)

    with open(outfile_tex, "w") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        
        f.write("\\begin{tabular}{c|>{\\columncolor{ibmorangelight}} r:>{\\columncolor{ibmpurplelight}}r|>{\\columncolor{ibmorangelight}}r:>{\\columncolor{ibmpurplelight}}r|>{\\columncolor{white}}r:>{\\columncolor{white}}r:>{\\columncolor{white}}r:>{\\columncolor{white}}r}\n")
        f.write("\\hline\n")
        f.write(" & ".join(out_fmt.columns) + " \\\\\n") 
        f.write("\\hline\n")

        for _, row in out_fmt.iterrows():
            line = " & ".join(str(x) for x in row.tolist()) + " \\\\\n"
            f.write(line)

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{UID values of \\textbf{Ground-XM3600} across languages under \\U{} and \\Pp{} conditions. "
        "The $\\Delta$ columns report the relative change in UID from \\U{} to \\Pp{}, "
        "computed as $\\frac{(\\text{\\Pp{}} - \\text{\\U{}})}{\\text{\\U{}}}$. "
        "$d_v$ and $d_{lv}$ report Cohen's $d$ standardized effect size for the paired difference. "
        "Bold values mark languages where UID is higher under \\Pp{}. "
        "Statistical significance is based on paired "
        "Wilcoxon signed-rank tests across sentences, with Benjamini--Hochberg FDR correction across languages. "
        "Significance levels are denoted as follows: $^\\ast$ $q<0.05$, $^{\\ast\\ast}$ $q<0.01$, $^{\\ast\\ast\\ast}$ $q<0.001$, "
        "n.s.~(not significant).}\n")
        f.write("\\label{tab:XCROSS_deltas}\n")
        f.write("\\end{table*}\n")

    print(f"Saved LaTeX table to {outfile_tex}")

    os.makedirs("data/XCROSS_processed_data", exist_ok=True)
    df_all = pd.DataFrame(all_sentence_uids)
    df_all.to_parquet("data/XCROSS_processed_data/XCROSS_uid_data.parquet", index=False)
    print("Saved UID data to data/XCROSS_processed_data/XCROSS_uid_data.parquet")

    return pd.DataFrame(all_sentence_uids)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='data/groundedness.csv', help="Path to caption CSV")
    parser.add_argument("--summary", action="store_true", help="Generate LaTeX table with dataset statistics.")
    parser.add_argument("--dataset", default="xm", help="Dataset flag in CSV")
    parser.add_argument("--out", default="latex/tabs/APP_XCROSS_deltas.tex", help="LaTeX output file with UID values.")

    args = parser.parse_args()

    cache_path = "data/XCROSS_processed_data/XCROSS_uid_data.parquet" #speed up plot test
    if os.path.exists(cache_path):
        print(f"Loading cached UID data from {cache_path}")
        all_data = pd.read_parquet(cache_path)
    else:
        all_data = main(csv_path=args.csv, #compute UID, perform wilcoxon test, save things.
                        dataset=args.dataset,
                        outfile_tex=args.out,
                        min_len=3,
                        do_summary=args.summary)


    plot_uid_scatter_from_table(
        uid_table_path=args.out,
        output_pdf="latex/imgs/pics/XCROSS_uid_scatterplots.pdf"
    )

    plot_global_uid_density(all_data, clip_percentile=99.0)
