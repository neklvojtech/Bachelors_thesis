# =============================================================================
# File: experiment_ALS.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 13.3.2025
# Description: Script to run ALS matrix factorization.
# Notes: Modified as part of the bachelor's thesis work. Evaluation changed to user-split.
# =============================================================================

import os
import argparse

# fix this somehow in the future
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
parser.add_argument("--device", default=None, type=str, help="Limit device to run on, default None (no limit)")

parser.add_argument("--dataset", default="", type=str, help="dataset. For list of datasets leave this parameter blank.")
parser.add_argument("--validation", default=False, type=str2bool, help="Use validation split: true/false")

parser.add_argument("--pu", default=1, type=int, help="User pruning aplied on training data.")
parser.add_argument("--pi", default=1, type=int, help="Item pruning aplied on training data.")

parser.add_argument("--factors", default=1024, type=int, help="ALS mf factors")
parser.add_argument("--regularization", default=50, type=float, help="ALS mf regularization")
parser.add_argument("--iterations", default=20, type=int, help="ALS mf iterations")
parser.add_argument("--num_threads", default=0, type=int, help="ALS mf num threads to use")
parser.add_argument("--use_gpu", default=False, type=str2bool, help="ALS mf whether to use gpu")


parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")

args = parser.parse_args([] if "__file__" not in globals() else None)

if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    
from _datasets.utils import *
from baselines import ALSMatrixFactorizer

from config import config

from time import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")
    
if __name__ == "__main__":
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)
    torch.manual_seed(args.seed)
    #keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    try:
        assert args.dataset in config.keys()
    except:
        print(f"Dataset must be one of {list(config.keys())}.")
        raise
    
    dataset, params = config[args.dataset]
    params['random_state'] = args.seed
    print(f"Loding dataset {args.dataset} with params {params}")
    dataset.load_interactions(**params)
    print(dataset)
    
    if args.validation:
        print("creating validation evaluator")
        val_evaluator = Evaluation(dataset, "validation")
        df = fast_pruning(dataset.train_interactions, args.pu,args.pi)
    else:
        df = fast_pruning(dataset.full_train_interactions, args.pu,args.pi)
    
    X = get_sparse_matrix_from_dataframe(df)
    
    print(f"Interaction matrix: {repr(X)}")
    
    print("creating test evaluator")
    test_evaluator = Evaluation(dataset, "test")
    
    print()
    
    model = ALSMatrixFactorizer(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        use_gpu=args.use_gpu,
        num_threads=args.num_threads,
        item_idx=df.item_id.cat.categories,
        #item_idx=dataset.full_train_interactions.item_id.cat.categories, 
    )
    
    fits = []
    val_logs = []
    start = time()
    model.fit(X)
    train_time=start-time()
    if args.validation:
        val_df_preds = model.predict_df(val_evaluator.test_src)
        val_results=val_evaluator(val_df_preds)
        val_logs.append(val_results)
        dff = pd.DataFrame(val_logs)
        dff["epoch"] = np.arange(dff.shape[0])+1
        dff[list(dff.columns[-1:])+list(dff.columns[:-1])]
        dff.to_csv(f"{folder}/val_logs.csv")
        print("val_logs file written")

    df_preds = model.predict_df(test_evaluator.test_src)
    results=test_evaluator(df_preds)
    
    print(results)
    
    df = pd.DataFrame()
    
    df.to_csv(f"{folder}/history.csv")
    print("history file written")
    
    pd.Series(results).to_csv(f"{folder}/result.csv")
    print("results file written")
    
    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")