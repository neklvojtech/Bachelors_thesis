# =============================================================================
# File: content_based_similarity.py
# Description: This script runs a CBF model using audio embeddings or attributes.
# Author: Vojtěch Nekl
# Created: 9.1.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================

import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import numpy as np
import torch
import pandas as pd
import time
import keras

from utils import *
from Audio_processor.Attributes_processor import Attributes_Processor, Embedding_Processor
from config import config
from models import SparseKerasELSA, EmbeddingSimilarityModel

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
parser.add_argument("--device", default=None, type=str, help="Limit device to run on")
# dataset
parser.add_argument("--dataset", default="audiodataset_audio", type=str, help="Dataset to run on")
parser.add_argument("--validation", default=False, type=str2bool, help="Use validation split: true/false")
parser.add_argument("--use_cold_start", default=False, type=str2bool, help="Use cold start evaluation, default false")
parser.add_argument("--use_time_split", default=False, type=str2bool, help="Use time split evaluation, default false")

parser.add_argument("--audio_emb", default=False, type=str2bool, help="Do we want to use audio embeddings? [true/false]")
parser.add_argument("--standardize", default=True, type=str2bool, help="Standardize the input? [true/false]")

parser.add_argument("--audio_att", default=False, type=str2bool, help="Do we want to use audio attributes? [true/false]")
parser.add_argument("--pooling_time", default="end", type=str, help="Wheater to use attributes, that were created through mean of results, or through mean of embeddings and then classified. [end/emb]")
parser.add_argument("--which_attributes", default="all", type=str, choices=["all", "library", "ast"], help="Which attributes to use? Options: all, library, ast")
parser.add_argument("--scale_AST_att", default=False, type=str2bool, help="Scale AST attributes? [true/false]")
parser.add_argument("--standardize_ess_att", default=True, type=str2bool, help="Standardize essential attributes? [true/false]")
parser.add_argument("--standardize_ast_att", default=True, type=str2bool, help="Standardize ast attributes? [true/false]")

parser.add_argument("--similarity", default="cosine", type=str, help="Similarity measure to use (cosine, euclidean, dot)")

parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")

args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")


def main(args):
    train_time = 0
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    
    vargs["cuda_or_cpu"]=DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(folder)
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.dataset not in config.keys():
        print("Unknown dataset. List of available datasets: \n")
        for x in config.keys():
            print(x)
        return 

    dataset, params = config[args.dataset]
    dataset.load_interactions(**params)
    
    print(dataset)
        
    if args.use_time_split:
        evaluator_class = TimeBasedEvaluation
    elif args.use_cold_start:
        evaluator_class = ColdStartEvaluation
    else:  # Default: user-based split strategy
        evaluator_class = Evaluation
        
    test_evaluator = evaluator_class(dataset, "test")
        
    if args.validation:
        val_evaluator = evaluator_class(dataset, "validation")
    else:
        val_evaluator = None

    print(dataset)

    if args.audio_emb:
        preprocessing_config = {
            "embeddings_json_path": dataset.embeddings_json_path,
            "standardize": args.standardize
        }
        processor = Embedding_Processor(preprocessing_config, dataset.train_interactions.item_id.cat.categories)
        embs_full = processor.get_ordered_tensor(dataset.all_interactions.item_id.cat.categories)

    elif args.audio_att:
        if args.pooling_time not in {"end", "emb"}:
            raise ValueError("Pooling time must be either 'end' or 'emb'")

        preprocessing_config = {
            "csv_path": dataset.csv_path_end_pooling if args.pooling_time == "end" else dataset.csv_path_emb_pooling,
            "pooling_time": args.pooling_time,
            "AST_occurence_path": dataset.AST_occurence_path,
            "standardize_ess_att": args.standardize_ess_att,
            "standardize_ast_att": args.standardize_ast_att,
            "scale_AST_att": args.scale_AST_att,
            "which_attributes": args.which_attributes
        }
        
        processor = Attributes_Processor(preprocessing_config, dataset.train_interactions.item_id.cat.categories)
        embs_full = processor.get_ordered_tensor(dataset.all_interactions.item_id.cat.categories)

    else:
        raise ValueError("Choose which data should be used for evaluation (attributes/embeddings)")
          
    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("Similarity")
    model = EmbeddingSimilarityModel(
        item_embeddings=torch.from_numpy(embs_full), 
        items_idx=dataset.all_interactions.item_id.cat.categories,
        similarity=args.similarity,
        device=DEVICE
    )       

    val_logs = []
    if args.validation:
        val_df_preds = model.predict_df(val_evaluator.test_src)
        val_results=val_evaluator(val_df_preds)
        val_logs.append(val_results)
        dff = pd.DataFrame(val_logs)
        dff["epoch"] = np.arange(dff.shape[0])+1 # always just "one epoch", but staying consistent with other models
        dff[list(dff.columns[-1:])+list(dff.columns[:-1])]
        dff.to_csv(f"{folder}/val_logs.csv")
        print("val_logs file written")
    
    print("Calculating predictions...")
    
    if args.use_cold_start:
        df_preds = model.predict_df(test_evaluator.test_src, candidates_df=test_evaluator.candidates_df)
    else:
        df_preds = model.predict_df(test_evaluator.test_src)
        
    results = test_evaluator(df_preds)
    print(results)

    # final logs
    pd.Series(results).to_csv(f"{folder}/result.csv")
    print("results file written")

    train_time = time.time() - train_time
    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")

if __name__ == "__main__":
    main(args)



