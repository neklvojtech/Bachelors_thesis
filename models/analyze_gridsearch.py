# =============================================================================
# File: analyze_gridsearch.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 3.4.2025
# Description: Analyzes the results of a grid search for different models and configurations.
# Notes: Modified as part of the bachelor's thesis work. Only small adjustments were made to the original code.
# =============================================================================

import analyzer
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

parser = argparse.ArgumentParser()

parser.add_argument("-d","--dir", default="results", type=str, help="Dir where the results are stored")
parser.add_argument('-f','--flags', nargs='+', help='Set flag', default=[])
parser.add_argument('-g','--groupby', nargs='+', help='Set groupbys', default=[])
parser.add_argument('-m','--mean', nargs='+', help='Set means', default=[])
parser.add_argument('-s','--std', nargs='+', help='Set stds', default=[])
parser.add_argument('-q','--query', default="seed==seed",type=str, help="query for limiting the results")
parser.add_argument('-t','--dataset', nargs='+', default=["audiodataset_audio"],type=str, help="Evaluated dataset.")
parser.add_argument('-o','--sort', nargs='+', default=["val_ndcg@50"],type=str, help="Sort by.")
parser.add_argument("-v","--save", default="none", type=str, help="file to save as csv for later analysis")


args = parser.parse_args([] if "__file__" not in globals() else None)

#print(args.flags)
#print(type(args.flags))

data = analyzer.get_raw_data(args.dir)

if len(args.groupby)==0:
    groupby = ["flag",'factors', 'batch_size', 'epochs','max_output']
else:
    groupby = args.groupby

if len(args.mean)==0:
    mean = ["val_ndcg@50", "val_recall@5","val_recall@10","val_recall@20", "val_coverage@20", "ndcg@50", "recall@5","recall@10","recall@20", "coverage@20"]
else:
    mean = args.mean

if len(args.std)==0:
    std = ["val_recall@5","val_recall@10","val_recall@20", "val_ndcg@50", "val_coverage@20"]
else:
    std = args.std

if len(args.flags)==0:
    flags = ["ease_test", "mf_test", "knn_test", "top_popular_test", "sparse_elsa_test", "dense_elsa_test"]
else:
    flags = args.flags
#print(data)
data = data[data.dataset.isin(args.dataset)]
data = data.query(args.query)

data = data.infer_objects(copy=False).fillna(0)
#print(data)
cols = data.columns
#print(cols)
#data[cols] = data[cols].apply(pd.to_numeric, errors='ignore')

res_test1 = data[data.flag.isin(flags)].groupby(groupby)[mean].mean().reset_index()
res_test2 = data[data.flag.isin(flags)].groupby(groupby)[std].std().reset_index()[std]
res_test2.columns = [x.replace("val","std") if "val" in x else f"std_{x}" for x in std]

data[data.flag.isin(flags)].to_csv("debug.csv")

res_test3 = data[data.flag.isin(flags)].groupby(groupby)[mean].size().reset_index().iloc[:,-1:]
res_test3.columns=["n_experiments"]
pd.options.display.max_columns = None

print("Results:")
pd.set_option('display.max_rows', None)  
df = pd.concat([res_test1, res_test2, res_test3], axis=1).sort_values(args.sort, ascending=False)
print(df)
if args.save!="none":
    df.to_csv("experiments_results/" + args.save + ".csv")

    def round_df(df, sig_digits=5):
        df_rounded = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_rounded[col] = df[col].apply(lambda x: f"{x:.5g}")
        return df_rounded

    # --- Your existing code with tweaks ---
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    font_size = 12
    row_height = 0.4

    # Round numerical values
    df_display = round_df(df)

    # Create figure
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.2, len(df) * row_height + 1))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='left',  # Default cell alignment
        loc='center',
        colColours=['#f2f2f2'] * df_display.shape[1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # Adjust column widths
    col_widths = [max([len(str(s)) for s in df_display[col].values] + [len(col)]) for col in df_display.columns]
    col_widths = [w / max(col_widths) for w in col_widths]  # normalize

    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)
                cell.set_text_props(ha='left', va='center')  # Align left

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='black', ha='center')
            cell.set_facecolor('#e0e0e0')
        else:
            cell.set_facecolor('#ffffff' if row % 2 == 0 else '#f9f9f9')

        cell.set_edgecolor('#dddddd')

    # Save image
    plt.tight_layout()
    plt.savefig("experiments_results/" + args.save, bbox_inches='tight', dpi=300)
    plt.close()