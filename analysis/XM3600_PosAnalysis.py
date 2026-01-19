# generates:
# - Figure 6
# - Figure 7

import polars as pl

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import numpy as np

import os

matplotlib.use("pgf")
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

fontsize = 15
matplotlib.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "axes.labelsize": fontsize + 1,
    "axes.titlesize": fontsize + 2,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "pgf.preamble": r"""
        \PassOptionsToPackage{table,x11names}{xcolor}
        \usepackage{xcolor}
        \usepackage{mathptmx}
        \usepackage{amsmath}
    """
})


## PLOTTING

def plot_pos_heatmap(
    csv_path: str = "data/XCROSS_processed_data/POS_analysis.csv",
    out_dir: str = "latex/imgs/pics"
):
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Please run the decomposition script first.")
        return

    df = pd.read_csv(csv_path)
    
    valid_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'PRON', 'NUM', 'ADP', 'DET', 'AUX']
    df = df[df['pos'].isin(valid_pos)]
    

    metric_col = "avg_prevention_score"

    heatmap_data = df.pivot(
        index="pos", 
        columns="lang", 
        values=metric_col
    )
    
    pos_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index
    heatmap_data = heatmap_data.loc[pos_order]
    
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

    width = max(6, len(heatmap_data.columns) * 0.6) 
    fig, ax = plt.subplots(figsize=(width, 6.0))
    
    v_clean = heatmap_data.values[~np.isnan(heatmap_data.values)]
    vlim = max(abs(np.percentile(v_clean, 5)), abs(np.percentile(v_clean, 95)))
    
    sns.heatmap(
        heatmap_data, 
        cmap="vlag", 
        center=0.0, 
        vmin=-vlim, 
        vmax=vlim, 
        annot=True, 
        fmt=".2f", 
        linewidths=0.5, 
        cbar_kws={"label": r"$\overline{\Delta C}_{\text{POS}}$"},
        ax=ax,
        annot_kws={"size": 8},
        square=True,
        robust=True
    )
    
    #ax.set_title(r"\textbf{Responsibility for UID Failure by POS (P - U)}")
    ax.set_xlabel("Language")
    ax.set_ylabel("Part-of-Speech")
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    heatmap_path = os.path.join(out_dir, "XCROSS_heatmap_distribution.pdf")
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    
    print(f"Heatmap saved to {heatmap_path}")

def plot_pos_6x5(
    csv_path: str = "data/XCROSS_processed_data/POS_analysis.csv",
    out_path: str = "latex/imgs/pics/XCROSS_pos.pdf"
):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_all = pd.read_csv(csv_path)
    
    valid_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'PRON', 'NUM', 'ADP', 'DET', 'AUX']
    df_all = df_all[df_all['pos'].isin(valid_pos)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    languages = sorted(df_all['lang'].unique())
    rows, cols = 6, 5
    

    fig = plt.figure(figsize=(18, 24))
    
    for i, lang in enumerate(languages):
        if i >= rows * cols: break
        
        ax1 = fig.add_subplot(rows, cols, i + 1)
        

        df_lang = df_all[df_all['lang'] == lang].sort_values("avg_prevention_score", ascending=False)
        if df_lang.empty: continue
            
        pos = df_lang['pos']
        scores = df_lang['avg_prevention_score']
        

        ax1.bar(pos, scores, alpha=0.15, color='black', edgecolor='black', 
            label=r"$\overline{\Delta C}_{\text{POS}}$")
        ax1.set_title(rf"\textbf{{{lang.upper()}}}", fontsize=16, pad=10)
        
 
        ax1.tick_params(axis='x', labelsize=8, rotation=45)
        ax1.tick_params(axis='y', labelsize=10)
        

        ax2 = ax1.twinx()
        ax2.plot(pos, df_lang['perc_increase'] * 100, marker='o', color='#D22B2B', 
                 linewidth=1.2, markersize=4, label=r'\% Surprisal Increase')
        ax2.plot(pos, df_lang['perc_decrease'] * 100, marker='s', color='#4C72B0', 
                 linewidth=1.2, linestyle='--', markersize=4, label=r'\% Surprisal Decrease')
        
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y', labelsize=10)
        

        if i % cols == 0:
            ax1.set_ylabel(r"$\overline{\Delta C}_{\text{POS}}$", fontsize=13, fontweight='bold')
        if (i + 1) % cols == 0:
            ax2.set_ylabel(r"\% of Words in POS", fontsize=13, fontweight='bold')
        else:
            ax2.set_yticklabels([])

        ax1.grid(axis='y', linestyle=':', alpha=0.4)


    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 0.985), 
               ncol=3, fontsize=16, frameon=False)


    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.04, wspace=0.18, hspace=0.35)
    
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"6x5 Grid saved to {out_path}")


#########

#########

# ANALYSE POS

def POS_uid_decomposition(
    csv_path: str,
    dataset: str = "xm",
    out_dir: str = "data/XCROSS_processed_data"
):
    os.makedirs(out_dir, exist_ok=True)

    q = (
        pl.scan_csv(csv_path)
        .filter(pl.col("dataset") == dataset)
        .with_columns([
            (-1 * pl.col("log_lm").cast(pl.Float64)).alias("surp_U"),
            (-1 * pl.col("log_cap").cast(pl.Float64)).alias("surp_P"),
            pl.col("sentence_idx").cast(pl.Int64),
            pl.col("word").fill_null("<UNK>")
        ])
    )

    columns = q.limit(0).collect().columns
    pos_col = next((c for c in ["pos", "upos", "pos_tag", "POS"] if c in columns), None)
    if not pos_col: raise ValueError("No POS column found")
    q = q.rename({pos_col: "pos"})

    df = q.collect()
    df = df.with_columns([
        (pl.col("sentence_idx") == 0).cum_sum().alias("global_sentence_id")
    ])

    # C_t = (s_t - mu)^2 / n
    df_calc = df.with_columns([
        pl.col("surp_U").mean().over("global_sentence_id").alias("mean_U"),
        pl.col("surp_P").mean().over("global_sentence_id").alias("mean_P"),
        pl.len().over("global_sentence_id").alias("n_tokens")
    ]).with_columns([
        ((pl.col("surp_U") - pl.col("mean_U")).pow(2) / pl.col("n_tokens")).alias("contrib_U"),
        ((pl.col("surp_P") - pl.col("mean_P")).pow(2) / pl.col("n_tokens")).alias("contrib_P")
    ])

    # Delta 
    df_calc = df_calc.with_columns([
        (pl.col("contrib_P") - pl.col("contrib_U")).alias("delta_contrib")
    ])


    sent_summ = df_calc.group_by(["lang", "global_sentence_id"]).agg([
        pl.col("contrib_U").sum().alias("uid_U"),
        pl.col("contrib_P").sum().alias("uid_P")
    ])

    df_analyzed = df_calc.join(sent_summ, on=["lang", "global_sentence_id"])

    # Aggregate by POS

    global_drivers = (
        df_analyzed.filter(pl.col("uid_P") > pl.col("uid_U")) # Isolate failure sentences
        .group_by(["lang", "pos"])
        .agg([

            pl.col("delta_contrib").mean().alias("avg_prevention_score"),
            
            (pl.col("surp_P") > pl.col("surp_U")).mean().alias("perc_increase"),

            (pl.col("surp_P") < pl.col("surp_U")).mean().alias("perc_decrease"),
            
            pl.len().alias("count")
        ])
        .filter(pl.col("count") > 50)
        .sort(["lang", "avg_prevention_score"], descending=True)
    )

    out_file = os.path.join(out_dir, "POS_analysis.csv")
    global_drivers.write_csv(out_file)
    print(f"Data saved to {out_file}")


if __name__ == "__main__":
    if not os.path.exists('data/XCROSS_processed_data/POS_analysis.csv'):
        POS_uid_decomposition("data/groundedness.csv")
    plot_pos_heatmap()
    plot_pos_6x5()
