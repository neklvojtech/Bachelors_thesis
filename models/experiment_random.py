# =============================================================================
# File: experiment_ranodm.py
# Description: Script to run Random Recommender model.
# Author: Vojtěch Nekl
# Created: 14.3.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================


import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'true', '1'):
        return True
    elif v.lower() in ('False', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--dataset", default="", type=str, help="Dataset. Leave blank for a list of datasets.")
parser.add_argument("--validation", default=False, type=str2bool, help="Use validation split: true/false")
parser.add_argument("--pu", default=5, type=int, help="User pruning applied on training data.")
parser.add_argument("--pi", default=1, type=int, help="Item pruning applied on training data.")
parser.add_argument("--flag", default="none", type=str, help="Flag for distinction of experiments, default none")

args = parser.parse_args([] if "__file__" not in globals() else None)


import numpy as np
import pandas as pd
import torch

from _datasets.utils import *
from baselines import RandomRecommender
from config import config

from time import time

if __name__ == "__main__":
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6) + np.random.randint(999999)}".replace(" ", "_")
    os.makedirs(folder, exist_ok=True)
    
    vargs = vars(args)
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(args)
    
    try:
        assert args.dataset in config.keys()
    except:
        print(f"Dataset must be one of {list(config.keys())}.")
        raise
    
    dataset, params = config[args.dataset]
    params['random_state'] = args.seed
    print(f"Loading dataset {args.dataset} with params {params}")
    dataset.load_interactions(**params)
    print(dataset)
    
    if args.validation:
        print("Creating validation evaluator")
        val_evaluator = Evaluation(dataset, "validation")
        df = fast_pruning(dataset.train_interactions, args.pu, args.pi)
    else:
        df = fast_pruning(dataset.full_train_interactions, args.pu, args.pi)
    
    X = get_sparse_matrix_from_dataframe(df)
    
    print(f"Interaction matrix: {repr(X)}")
    
    print("Creating test evaluator")
    test_evaluator = Evaluation(dataset, "test")
    
    print()
    
    model = RandomRecommender(item_idx=df.item_id.cat.categories, seed=args.seed)
    
    start = time()
    model.fit(X)
    train_time = time() - start
    
    if args.validation:
        val_df_preds = model.predict_random(val_evaluator.test_src)
        val_results = val_evaluator(val_df_preds)
        val_logs = [val_results]
        
        dff = pd.DataFrame(val_logs)
        dff["epoch"] = np.arange(dff.shape[0]) + 1
        dff.to_csv(f"{folder}/val_logs.csv")
        print("val_logs file written")
    
    df_preds = model.predict_random(test_evaluator.test_src)
    results = test_evaluator(df_preds)
    
    print(results)
    
    df = pd.DataFrame()
    df.to_csv(f"{folder}/history.csv")
    print("history file written")
    
    pd.Series(results).to_csv(f"{folder}/result.csv")
    print("results file written")
    
    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")
