# generates: 
# - Table 4 (Appendix)

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import os
from tqdm import tqdm
from utils import iso_639_3



def generate_latex_table(df_res: pd.DataFrame, outfile_tex: str):
    #latex table    
    def get_stars(q):
        if q < 0.001: return '^{***}'
        if q < 0.01: return '^{**}'
        if q < 0.05: return '^{*}'
        return '^{ns}'

    def fmt_delta(val, q):
        txt = f"{val:.2f}"
        if val > 0: txt = f"\\textbf{{{txt}}}"
        return f"${txt}{get_stars(q)}$"

    out = pd.DataFrame(index=df_res.index)
    out["Lang"] = df_res.index
    out["g_u"] = df_res["g_u"].apply(lambda x: f"{x:.2f}")
    out["g_p"] = df_res["g_p"].apply(lambda x: f"{x:.2f}")
    out["g_d"] = [fmt_delta(r.d_glob, r.q_glob) for r in df_res.itertuples()]
    out["l_u"] = df_res["l_u"].apply(lambda x: f"{x:.2f}")
    out["l_p"] = df_res["l_p"].apply(lambda x: f"{x:.2f}")
    out["l_d"] = [fmt_delta(r.d_loc, r.q_loc) for r in df_res.itertuples()]

    os.makedirs(os.path.dirname(outfile_tex), exist_ok=True)
    with open(outfile_tex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{3.5pt}\n")
        f.write("\\begin{tabular}{l|rr>{\\columncolor{gray!10}}r|rr>{\\columncolor{gray!10}}r}\n")
        f.write("\\toprule\n")
        f.write(r"\textbf{Lang} & \multicolumn{3}{c}{\textbf{Global CV}} & \multicolumn{3}{c}{\textbf{Local CV}} \\" + "\n")
        f.write(r" & \U{} & \Pp{} & $\Delta\%$ & \U{} & \Pp{} & $\Delta\%$ \\" + "\n")
        f.write("\\midrule\n")

        for _, row in out.iterrows():
            f.write(f"{row.Lang} & {row.g_u} & {row.g_p} & {row.g_d} & {row.l_u} & {row.l_p} & {row.l_d} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Mean Global CV (Coefficient of Variation) and Local CV across languages and conditions for captions in \\textsc{GROUND-XM3600}.}\n")
        f.write("\\label{tab:xcross_coeffvar}\n\\end{table}\n")


#compute
def coefficient_variation(log_probs: np.ndarray, local: bool = False) -> float:
    if len(log_probs) < 2:
        return np.nan
    
    mu = np.mean(log_probs)
    if mu <= 1e-9: 
        return np.nan

    if local:
        diffs = np.diff(log_probs)
        rmssd = np.sqrt(np.mean(diffs ** 2))
        return rmssd / mu
    else:
        sigma = np.std(log_probs, ddof=1)
        return sigma / mu


########
########
########


def main(csv_path: str, dataset: str, outfile_tex: str, min_len: int = 3):
    df = (pl.read_csv(csv_path)
          .filter(pl.col("dataset") == dataset)
          .with_columns(pl.col("log_lm").cast(pl.Float64),
                        pl.col("log_cap").cast(pl.Float64)))

    rows = []
    for lang in tqdm(df["lang"].unique().to_list(), desc="Processing"):
        lang_df = df.filter(pl.col("lang") == lang)
        
        # build sentences (0 =start new sentence)
        idxs = lang_df.select(pl.arg_where(pl.col("sentence_idx") == 0)).to_series().to_list() + [len(lang_df)]
        
        m_lm_glob, m_lm_loc, m_cap_glob, m_cap_loc = [], [], [], []
        
        for i in range(len(idxs)-1):
            if idxs[i+1] - idxs[i] < min_len: continue
            s = lang_df[idxs[i]:idxs[i+1]]
            
            lm, cap = np.array(s["log_lm"]), np.array(s["log_cap"])
            
            m_lm_glob.append(coefficient_variation(lm, local=False))
            m_lm_loc.append(coefficient_variation(lm, local=True))
            m_cap_glob.append(coefficient_variation(cap, local=False))
            m_cap_loc.append(coefficient_variation(cap, local=True))

        valid_g = [(x, y) for x, y in zip(m_lm_glob, m_cap_glob) if not (np.isnan(x) or np.isnan(y))]
        valid_l = [(x, y) for x, y in zip(m_lm_loc, m_cap_loc) if not (np.isnan(x) or np.isnan(y))]
        
        lg, cg = zip(*valid_g)
        ll, cl = zip(*valid_l)

        _, p_glob = wilcoxon(lg, cg)
        _, p_loc = wilcoxon(ll, cl)

        rows.append({
            "lang": iso_639_3.get(lang, lang),
            "g_u": np.mean(lg), "g_p": np.mean(cg),
            "l_u": np.mean(ll), "l_p": np.mean(cl),
            "d_glob": 100 * (np.mean(cg) - np.mean(lg)) / np.mean(lg),
            "d_loc": 100 * (np.mean(cl) - np.mean(ll)) / np.mean(ll),
            "p_glob": p_glob, "p_loc": p_loc
        })

    df_res = pd.DataFrame(rows).set_index("lang").sort_index()
    for m in ["glob", "loc"]:
        _, qvals, _, _ = multipletests(df_res[f"p_{m}"], alpha=0.05, method="fdr_bh")
        df_res[f"q_{m}"] = qvals

    generate_latex_table(df_res, outfile_tex)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='data/groundedness.csv')
    parser.add_argument("--dataset", default="xm")
    parser.add_argument("--out", default="latex/tabs/APP_XCROSS_coeffvar.tex")
    args = parser.parse_args()
    main(args.csv, args.dataset, args.out)