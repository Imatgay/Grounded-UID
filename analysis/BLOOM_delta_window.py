# Generates:
# - Table 8
# - Table 9

import pandas as pd, numpy as np, ast
from pathlib import Path

def export_latex_table(df_diff, unit_label, tex_file_name, caption_label, LATEX_DIR):
    pivot = df_diff.pivot_table(index=["lang", "window"], columns="cond", values="diff", aggfunc="mean")
    pivot = pivot[["T", "P", "D", "DP"]]  
    pivot = pivot.round(3).reset_index()

    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\renewcommand{\arraystretch}{1.2}")
    latex_lines.append(r"\begin{tabular}{c|c|r:r:r:r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Lang} & $\Delta_w$ & \U & \Pp & \D & $[\text{\Pp} + \text{\D}]$ \\")
    latex_lines.append(r"\midrule")

    for lang in pivot["lang"].unique():
        lang_rows = pivot[pivot["lang"] == lang]
        for i, row in lang_rows.iterrows():
            delta_w = int(row["window"])
            vals = [f"{row[c]:.3f}" if pd.notnull(row[c]) else "–" for c in ["T", "P", "D", "DP"]]
            if i == lang_rows.index[0]:
                prefix = f"\\multirow{{3}}{{*}}{{{lang}}} & $\\Delta_{{{delta_w}}}$"
            else:
                prefix = f"& $\\Delta_{{{delta_w}}}$"
            latex_lines.append(f"{prefix} & " + " & ".join(vals) + r" \\")
        latex_lines.append(r"\cline{1-6}")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(
        rf"\caption{{Mean surprisal difference across \textbf{{paragraph}} boundaries ($\Delta_w$) for window sizes $w = 1, 2, 3$, "
        rf"reported per language. For each transition, we subtract the average surprisal of the final $w$ words of a paragraph "
        rf"from that of the first $w$ words of the following one. More negative values indicate larger spikes in surprisal at "
        rf"paragraph onsets, reflecting greater non-uniformity in information flow across discourse boundaries."
    )
    latex_lines.append(rf"\label{{tab:directional_window_deltas_{caption_label}}}")
    latex_lines.append(r"\end{table*}")

    tex_path = LATEX_DIR / tex_file_name
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"LaTeX table written to {tex_path}")

def extract_directional_difference(df, unit):
    records = []
    for w in [1, 2, 3]:
        col_back = f"backward_window_{w}"
        col_start = f"starting_window_{w}"
        if col_back not in df or col_start not in df:
            continue
        for r in df.itertuples():
            try:
                lst_back = ast.literal_eval(getattr(r, col_back))
                lst_start = ast.literal_eval(getattr(r, col_start))
                if lst_back and lst_start:
                    diff = np.mean(lst_back) - np.mean(lst_start)
                    records.append(dict(
                        lang=r.lang,
                        cond=r.cond,
                        window=w,
                        diff=diff,
                        unit=unit
                    ))
            except Exception:
                continue
    return pd.DataFrame(records)


def main():
    IN_DIR = Path("data/BLOOM_processed_data")
    OUT_DIR = Path("data/_other_data")
    LATEX_DIR = Path("latex/tabs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    df_para = pd.read_csv(IN_DIR / "paragraph_metrics.csv")
    df_sent = pd.read_csv(IN_DIR / "sentence_metrics.csv")


    df_diff_para = extract_directional_difference(df_para, "paragraph")
    df_diff_sent = extract_directional_difference(df_sent, "sentence")

    df_diff_all = pd.concat([df_diff_para, df_diff_sent])
    df_diff_all.to_csv(OUT_DIR / "window_directional_diff_by_cond_unit.csv", index=False)

    export_latex_table(df_diff_para, "paragraph", "APP_BLOOM_window_paragraph.tex", "paragraph", LATEX_DIR)
    export_latex_table(df_diff_sent, "sentence", "APP_BLOOM_window_sentence.tex", "sentence", LATEX_DIR)


if __name__ == "__main__":
    main()