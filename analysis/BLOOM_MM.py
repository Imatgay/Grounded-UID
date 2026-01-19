# generates:
# - Table 6 (Appendix)
# - Table 7 (APpendix)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path
import ast
import statsmodels.api as sm


IN_DIR = Path("data/BLOOM_processed_data")
OUT_LATEX_DIR = Path("latex")
LATEX_DIR = OUT_LATEX_DIR / "tabs"
OUT_DATA_DIR  = Path("data/_other_data")

LATEX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def export_latex_per_yvar(results_df, y_var, latex_dir):
    from pathlib import Path

    desired_order = ["T", "P", "D", "DP"]
    header_map = {
        "T": r"\U", "P": r"\Pp", "D": r"\D", "DP": r"[\text{\Pp} + \text{\D}]"
    }

    def format_sig(slope, p):
        if pd.isnull(p): return "–"
        if p < 0.001: sig = "***"
        elif p < 0.01: sig = "**"
        elif p < 0.05: sig = "*"
        else: sig = "n.s"
        return f"{slope:.2f}~{sig}"

    df = results_df[results_df["y_var"] == y_var].copy()
    
    df["entry"] = df.apply(lambda row: format_sig(row["slope"], row["pval"]), axis=1)


    table = df.pivot_table(
        index=["lang", "unit"],
        columns="condition",
        values="entry",
        aggfunc="first"
    ).sort_index()

    table = table.reindex(columns=desired_order)

    latex_path = Path(latex_dir) / f"APP_BLOOM_mixed_{y_var.lower()}.tex"
    
    with open(latex_path, "w") as f:
        f.write(r"\begin{table*}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.2}" + "\n")
        f.write(r"\begin{tabular}{c|c|r:r:r:r}" + "\n")
        f.write(r"\toprule" + "\n")
        
        header_row = r"\textbf{Lang} & \textbf{Unit} & " + " & ".join([header_map[c] for c in desired_order]) + r" \\" + "\n"
        f.write(header_row)
        f.write(r"\midrule" + "\n")

        languages = table.index.get_level_values('lang').unique()
        
        for lang in languages:
            sub_table = table.loc[lang]
            units = sub_table.index.tolist()
            num_units = len(units)

            for i, unit in enumerate(units):
                row_vals = " & ".join(sub_table.loc[unit].values)
                
                if i == 0:
                    lang_cell = f"\\multirow[t]{{{num_units}}}{{*}}{{{lang}}}"
                else:
                    lang_cell = ""
                
                f.write(f"{lang_cell} & {unit} & {row_vals} \\\\\n")
            
            f.write(r"\cline{1-6}" + "\n")


        var_display = r"$\bm{\mathrm{UID}}_{v}$" if y_var == "UIDv" else "Mean Surprisal"
        

        caption_text = (
            r"\caption{Fixed-effect slope estimates over relative position for " + var_display + 
            r", from mixed-effects models (with random effects by story), "
            r"computed separately per language and discourse unit (sentence, paragraph). "
            r"Slopes reflect condition-specific trajectories via interaction terms, with \U\ as the baseline. "
            r"Significance thresholds: $^\ast$ $p<0.05$, $^{\ast\ast}$ $p<0.01$, $^{\ast\ast\ast}$ $p<0.001$, n.s.~(not significant).}"
        )


        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(caption_text + "\n")
        f.write(f"\\label{{tab:BLOOM_mixed_{y_var.lower()}}}\n")
        f.write(r"\end{table*}" + "\n")

    print(f"Table written to: {latex_path.resolve()}")
def main():


    df_para = pd.read_csv(IN_DIR / "paragraph_metrics.csv")
    df_sent = pd.read_csv(IN_DIR / "sentence_metrics.csv")

    df_para["log_length"] = np.log1p(df_para["n_tokens"])
    df_sent["log_length"] = np.log1p(df_sent["n_tokens"])


    languages = sorted(df_para["lang"].unique())
    conditions = sorted(df_para["cond"].unique())

    sns.set(style="whitegrid", context="talk")


    from collections import defaultdict

    results = []


    ANALYSES = [
        ("paragraph", df_para, "rel_paragraph_pos"),
        ("sentence", df_sent, "rel_sent_pos")
    ]

    DEPENDENT_VARS = ["mean_surprisal", "UIDv"]

    results = []

    for unit, df, pos_col in ANALYSES:
        for lang in languages:
            sub = df[df.lang == lang].copy()
            if unit == "sentence":
                sub = sub.sort_values(["story_id", "paragraph", "sentence"])
                sub["abs_sent_pos"] = sub.groupby(["story_id", "paragraph"]).cumcount() + 1
                sub["log_pos"] = np.log1p(sub["abs_sent_pos"])  # log(1 + pos)
            elif unit == "paragraph":
                sub = sub.sort_values(["story_id", "paragraph"])
                sub["abs_par_pos"] = sub.groupby("story_id").cumcount() + 1
                sub["log_pos"] = np.log1p(sub["abs_par_pos"])
            
            #pos_col = "log_pos"

            if unit == 'sentence':
                sub = sub.groupby(["story_id", "paragraph"]).filter(lambda g: g["sentence"].nunique() >= 3)

            if sub["story_id"].nunique() < 5 or sub.shape[0] < 30:
                #print(f"[SKIP] Too little data for {lang} {unit}")
                continue

            sub["cond"] = pd.Categorical(sub["cond"], categories=["T", "P", "D", "DP"])

            for y_var in DEPENDENT_VARS:
                if y_var not in sub.columns:
                    continue

                formula = f"{y_var} ~ {pos_col} * C(cond) + log_length"
                re_form = f"~ {pos_col}"  

                try:
                    model = smf.mixedlm(
                        formula=formula,
                        data=sub,
                        groups=sub["story_id"],
                        re_formula=re_form
                    )
                    fit = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)

                    #print(f"[OK] {lang}-{unit}-{y_var} | re_formula = {re_form} | N_story = {sub['story_id'].nunique()} | N_obs = {sub.shape[0]}")

                except Exception as e:
                    print(f"[FAIL] {lang}-{unit}-{y_var}: {e}")
                    continue  

                fe = fit.params
                pvals = fit.pvalues



                base_coef = fe.get(pos_col, np.nan)
                base_pval = pvals.get(pos_col, np.nan)
                results.append({
                    "lang": lang,
                    "unit": unit,
                    "y_var": y_var,
                    "condition": "T",
                    "slope": base_coef,
                    "pval": base_pval,
                    "r2": np.nan,                
                    "re_formula": re_form
                })

                # interactions
                for cond in ["P", "D", "DP"]:
                    key = f"{pos_col}:C(cond)[T.{cond}]"
                    delta = fe.get(key, np.nan)
                    pval = pvals.get(key, np.nan)
                    slope = base_coef + delta if not np.isnan(delta) else np.nan
                    results.append({
                        "lang": lang,
                        "unit": unit,
                        "y_var": y_var,
                        "condition": cond,
                        "slope": slope,
                        "pval": pval,
                        "r2": np.nan,
                        "re_formula": re_form
                    })



    results_df = pd.DataFrame(results)

    dupes = results_df.duplicated(subset=["lang", "unit", "condition", "y_var"], keep=False)
    if dupes.any():
        print("[WARN] Duplicates found in regression results:")
        #print(results_df[dupes])

    results_df.drop_duplicates(subset=["lang", "unit", "condition", "y_var"], inplace=True)


    results_df.to_csv(OUT_DATA_DIR / "BLOOM_regression_results.csv", index=False)

    for y_var in DEPENDENT_VARS:
        export_latex_per_yvar(results_df, y_var, LATEX_DIR)


if __name__ == "__main__":
    main()