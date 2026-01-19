# Generates :
# - Table 5 (Appendix)

import os
import pandas as pd
import numpy as np

def save_latex_table(df_long, metric_col, outfile, caption, label):
    pivot = df_long.groupby(["language", "cond"])[metric_col].mean().unstack()
    
    desired_order = ["T", "P", "D", "DP"]
    cols = [c for c in desired_order if c in pivot.columns]
    pivot = pivot[cols]
    
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    with open(outfile, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        
        col_def = "l|" + "r" * len(cols)
        f.write(f"\\begin{{tabular}}{{{col_def}}}\n")
        f.write("\\toprule\n")
        

        headers_map = {
            "T": r"\U", 
            "P": r"\Pp", 
            "D": r"\D", 
            "DP": r"$[\text{\Pp} + \text{\D}]$"
        }
        header_cells = [headers_map.get(c, c) for c in cols]
        f.write(r"\textbf{Lang} & " + " & ".join(header_cells) + " \\\\\n")
        f.write("\\midrule\n")
        

        for lang, row in pivot.iterrows():
            vals = [f"{row[c]:.2f}" for c in cols]
            f.write(f"{lang} & " + " & ".join(vals) + " \\\\\n")
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table to {outfile}")

def main():
    csv_input_path = "data/BLOOM_processed_data/paragraph_metrics.csv"
    if not os.path.exists(csv_input_path):
        print(f"Error: {csv_input_path} not found.")
        return

    df = pd.read_csv(csv_input_path)

    df = df[~df["lang"].isin(["khm", "fil", "nep"])]

    mean_col = 'mean_surprisal'
    var_col = 'UIDv'
    
    if mean_col in df.columns and var_col in df.columns:
        df['UIDcv'] = np.sqrt(df[var_col]) / np.abs(df[mean_col])
        print("Successfully calculated UIDcv.")
    else:
        print("Error: Missing required columns for CV calculation.")
        return


    df_uidcv = df[["lang", "cond", "UIDcv"]].copy()
    df_uidcv.rename(columns={"lang": "language", "UIDcv": "uidcv"}, inplace=True)
    
    df_uidcv["language"] = pd.Categorical(
        df_uidcv["language"],
        categories=sorted(df_uidcv["language"].unique()),
        ordered=True
    )

    save_latex_table(
        df_uidcv, 
        metric_col="uidcv", 
        outfile="latex/tabs/APP_BLOOM_coeffvar.tex",
        caption=r"Mean Global CV across languages and conditions for paragraph in \textsc{BLOOM-Vist}.",
        label="tab:bloom_coeffvar"
    )

if __name__ == "__main__":
    main()