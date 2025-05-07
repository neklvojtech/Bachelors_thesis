# =============================================================================
# File: Attribute_extractor_config.py
# Description: Configuration for the feature extraction process.
# Author: Vojtěch Nekl
# Created: 28.04.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================


dataset_conf = {
    "audio_dir": "../models/_datasets/audiodataset/audiodataset-songs",
    
    # embeddings path determines behaviour of extractor -- if None, new embeddings would be calcualted
    #"embeddings_path": "../models/_datasets/audiodataset/9875_embeddings_60s__nopooling_distilled.json",
    "embeddings_path": None,
    "audio_conf": {
        'num_mel_bins': 128,
        'target_length': 1024,
        'sample_rate': 16000,
    },
    "metadata_csv": "../models/_datasets/audiodataset/recombee-audio-files-samples.csv",
    "GLOBAL_MEAN": -4.2677393,
    "GLOBAL_STD": 4.5689974,
    "model_name": "bookbot/distil-ast-audioset",
    "ontology_path": "ontology.json"
}