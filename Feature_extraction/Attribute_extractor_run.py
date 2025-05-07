# =============================================================================
# File: Attribute_extractor_run.py
# Description: This script runs the attribute extraction process for audio files.
# Author: Vojtěch Nekl
# Created: 28.04.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================


import os
import argparse

parser = argparse.ArgumentParser()
        
parser.add_argument("--seed", type=int, default=42, help="Random seed")  
parser.add_argument("--device", default=None, type=str, help="Limit device to run on, default None (no limit)")

parser.add_argument("--Only_generate_embeddings", type=bool, default=False, help="Only generate embeddings and then exit. If embeddings path already provided, nothing would happen.")
parser.add_argument("--spect", type=int, default=6, help="Number of spectograms per embedding")
parser.add_argument("--pooling", type=str, default=None, help="Pooling method to use ('max' or 'mean')") # if not choosen, all spect are stored

parser.add_argument("--thresholds", type=float, nargs='+', default=[0.01, 0.05, 0.05, 0.05], help="Thresholds for each mask")
parser.add_argument("--masks", type=str, nargs='+', default=["singing", "musical instrument", "music genre", "music mood"], help="List of masks to use")
parser.add_argument("--is_music", type=bool, default=True, help="Whether the model shoul determine if the audio is music or not")
parser.add_argument("--output_directory", type=str, default="../models/_datasets/audiodataset/", help="Directory to save result file in. (and embeddings, if generated)")
parser.add_argument("--output_file", type=str, default="results.csv", help="Output file to save results")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
parser.add_argument("--print", type=bool, default=False, help="Print results")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for DataLoader")
parser.add_argument("--item_ids_filter_csv", type=str, default=None, help="CSV file containing item IDs for attribute generation, default None ( no filter )")
parser.add_argument("--normalize", type=bool, default=True, help="Normalize audio before processing")

args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import Attribute_extractor_config

from Attribute_emb_name_dataset import EmbDataset
from Attribute_extractor import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

def main(args):
    vargs = vars(args)
    vargs["cuda_or_cpu"] = DEVICE
    
    folder = f"setup_attribute_extraction/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)

    torch.manual_seed(args.seed)
    dataset_conf = Attribute_extractor_config.dataset_conf
    

    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    pd.Series(dataset_conf).to_csv(f"{folder}/dataset_conf.csv")
    
    # specify subset of items to process
    item_ids = None
    if args.item_ids_filter_csv is not None:
        item_ids = pd.read_csv(args.item_ids_filter_csv)["music_id"].tolist()
        print(f"Filtering by {len(item_ids)} item IDs.")
    
    embeddings_path = dataset_conf["embeddings_path"]
    
    # If embedding path is not provided, compute embeddings
    if embeddings_path is None:
        # Create EmbeddingGenerator object
        embedding_generator = EmbeddingGenerator(
            dataset_conf["audio_dir"], dataset_conf["metadata_csv"], DEVICE, dataset_conf["audio_conf"],
            dataset_conf["model_name"], vargs["normalize"], dataset_conf["GLOBAL_MEAN"], dataset_conf["GLOBAL_STD"],
            vargs["spect"], vargs["pooling"], vargs["batch_size"], vargs["output_directory"] ,item_ids
        )
        print("No embeddings path provided.")
        embeddings_path = embedding_generator.generate_embeddings()

    if args.Only_generate_embeddings:
        print("Embeddings generated.")
        return
    
    # Create AttributeExtractor object
    extractor = AudioAttributeExtractor(dataset_conf, vargs, device=DEVICE)
    
    dataset = EmbDataset(
        dataset_conf["audio_dir"],
        dataset_conf["metadata_csv"],
        embeddings_path,
        item_ids
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)    

    print("Attributes generation")
    
    for i, (music_ids, file_paths, embeddings) in enumerate(tqdm(dataloader, desc="Processing batches")):
        is_first = (i == 0)
        extractor.forward(music_ids, file_paths, embeddings, is_first)
    
    #extractor.save_results_to_file()

if __name__ == "__main__":
    main(args)