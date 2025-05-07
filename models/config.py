# =============================================================================
# File: config.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 25.10.2024
# Description: Configuration for the recommender system models.
# Notes: Modified as part of the bachelor's thesis work. Configuration for audio dataset was added.
# =============================================================================


import os

os.environ["KERAS_BACKEND"] = "torch"

from utils import *

config = {
    "ml20m": (
        Dataset("MovieLens20M"),
        {
            "filename": "_datasets/ml20m/ratings.csv",
            "item_id_name": "movieId",
            "user_id_name": "userId",
            "value_name": "rating",
            "timestamp_name": "timestamp",
            "min_value_to_keep": 4.0,
            "user_min_support": 5,
            "set_all_values_to": 1.0,
            "num_test_users": 10000,
            "random_state": 42,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("_datasets/ml20m/item_text_descriptions.feather")""",
            "items_item_id_name": "movieId",
            "items_preprocess": """f'{row.llama31_description}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
        },
    ),
    "goodbooks": (
        Dataset("Goodbooks-10k"),
        {
            "raw_data": """pd.read_csv("_datasets/goodbooks/ratings.csv")""",
            "user_id_name": "user_id",
            "item_id_name": "book_id",
            "value_name": "rating",
            "min_value_to_keep": 4.0,
            "user_min_support": 5,
            "item_min_support": 1,
            "set_all_values_to": 1.0,
            "num_test_users": 2500,
            "random_state": 42,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("_datasets/goodbooks/item_text_descriptions.feather")""",
            "items_item_id_name": "book_id",
            "items_preprocess": """f'{row.llama31_description}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
        },
    ),
    "amazon-books": (
        Dataset("Amazon books"),
        {
            "raw_data": """pd.read_feather("_datasets/amazbooks/ratings.feather")""",
            "value_name": "rating",
            "item_id_name": "asin",
            "user_id_name": "uid",
            "timestamp_name": "timestamp",
            "min_value_to_keep": 4.0,
            "user_min_support": 5,
            "item_min_support": 1,
            "set_all_values_to": 1.0,
            "num_test_users": 10000,
            "random_state": 42,
            "max_steps": 1000,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("_datasets/amazbooks/item_text_descriptions.feather")""",
            "items_item_id_name": "asin",
            "items_preprocess": """f'{row.llama31_description}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
        },
    ),
    "amazon-electronics": (
        Dataset("Amazon electronics"),
        {
            "raw_data": """pd.read_feather("_datasets/amazon_electronics/electronics_interactions_small.feather")""",
            "value_name": "value",
            "item_id_name": "item_id",
            "user_id_name": "user_id",
            "timestamp_name": "timestamp",
            "min_value_to_keep": 1.0,
            "user_min_support": 1,
            "item_min_support": 1,
            "set_all_values_to": 1.0,
            "num_test_users": 10000,
            "random_state": 42,
            "max_steps": 1000,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("_datasets/amazon_electronics/electronics_items_small.feather")""",
            "items_item_id_name": "asin",
            "items_preprocess": """f'{row.title}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
            "images_dir": "_datasets/amazon_electronics/images"
        },
    ),
    "amazon-fashion": (
        Dataset("Amazon clothing, shoes and jewelry"),
        {
            "raw_data": """pd.read_feather("_datasets/amazon_fashion/fashion_interactions_small.feather")""",
            "value_name": "value",
            "item_id_name": "item_id",
            "user_id_name": "user_id",
            "timestamp_name": "timestamp",
            "min_value_to_keep": 1.0,
            "user_min_support": 1,
            "item_min_support": 1,
            "set_all_values_to": 1.0,
            "num_test_users": 10000,
            "random_state": 42,
            "max_steps": 1000,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("_datasets/amazon_fashion/fashion_items_small_text.feather")""",
            "items_item_id_name": "asin",
            "items_preprocess": """f'{row.text}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
            "images_dir": "_datasets/amazon_fashion/images"
        },
    ),
    "goodbooks-img": (
        Dataset("Goodbooks-10k"),
        {
            "raw_data": """pd.read_csv("_datasets/goodbooks/ratings.csv")""",
            "user_id_name": "user_id",
            "item_id_name": "book_id",
            "value_name": "rating",
            "min_value_to_keep": 4.0,
            "user_min_support": 5,
            "item_min_support": 1,
            "set_all_values_to": 1.0,
            "num_test_users": 2500,
            "random_state": 42,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("_datasets/goodbooks/item_text_descriptions_img.feather")""",
            "items_item_id_name": "book_id",
            "items_preprocess": """f'{row.llama31_description}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
            "images_dir": "_datasets/goodbooks/images",
        },
    ),
    "ml20m-img": (
        Dataset("MovieLens20M"),
        {
            "filename": "_datasets/ml20m/ratings.csv",
            "item_id_name": "movieId",
            "user_id_name": "userId",
            "value_name": "rating",
            "timestamp_name": "timestamp",
            "min_value_to_keep": 4.0,
            "user_min_support": 5,
            "set_all_values_to": 1.0,
            "num_test_users": 10000,
            "random_state": 42,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("_datasets/ml20m/item_text_descriptions_img.feather")""",
            "items_item_id_name": "movieId",
            "items_preprocess": """f'{row.llama31_description}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
            "images_dir": "_datasets/ml20m/images",
            "images_suffix": ".0",
        },
    ),
    # -------------------------------------------------------------------------------------------------------------------------------------
    "audiodataset_audio": (
        Dataset("AudioDataset"),
        {   
            # interactions
            "raw_data": """pd.read_feather("_datasets/audiodataset/audiodataset_aggregated_interactions_cleaned.feather")""",
            #"raw_data": """pd.read_feather("_datasets/audiodataset/audiodataset_aggregated_interactions_10%_sample.feather")""",            
            "item_id_name": "item_id",
            "user_id_name": "user_id",
            "value_name": "value",
            "timestamp_name": "timestamp",
            "user_min_support": 5,
            "item_min_support": 1,
            "set_all_values_to": 1.0,
            "num_test_users": 50000,
            "random_state": 42,
            "load_previous_splits": False,
            
            "metadata_path": "_datasets/audiodataset/recombee-audio-files-samples.csv",
            
            # audio_model -- config is based on AST
            "audio_prepr_config": {
                "audio_dir": "_datasets/audiodataset/audiodataset-songs",
                "sample_rate": 16000, # Sample rate for audio processing
                "normalize": True, # Whether to normalize audio
                "mean": -4.2677393,
                "std": 4.5689974,
                "target_length": 1024, # Target length of processed audio
                "num_mel_bins": 128, # Number of mel bins used in spectrogram    
            },
            # audio attributes model
            "AST_occurence_path": "_datasets/audiodataset/label_counter.json",
            "csv_path_end_pooling": "_datasets/audiodataset/final_attributes_end_mean_pooling.csv", # one of paths is used based on hyperparameter selection
            "csv_path_emb_pooling": "_datasets/audiodataset/final_attributes_emb_mean_pooling.csv",
        
            # audio embeddings
            "embeddings_json_path": "_datasets/audiodataset/9876_embeddings_30s_mean_distilled.json",
            #"embeddings_json_path": "_datasets/audiodataset/9875_embeddings_60s_mean_distilled.json",
        }
    ),
}