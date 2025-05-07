# =============================================================================
# File: Attribute_emb_name_dataset.py
# Description: Dataset for loading audio embeddings from a directory.
# Author: Vojtěch Nekl
# Created: 28.04.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================


import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

class EmbDataset(Dataset):
    def __init__(self, audio_dir, metadata_csv, embeddings_path, item_ids=None):
        """
        Dataset class for loading audio embeddings from a directory.
        :param audio_dir: Directory containing audio files.
        :param metadata_csv: Path to a CSV file containing metadata about the audio files.
        :param embeddings_path: Path to a JSON file containing the embeddings.
        :param item_ids: List of item IDs to load. If None, all items will be loaded
        """
        self.audio_dir = audio_dir
        self.metadata = pd.read_csv(metadata_csv)
        self.load_embeddings(embeddings_path)
        self.filter_data(item_ids)
        
    def load_embeddings(self, embeddings_path):
        print(f"Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'r') as f:
            self.embeddings = json.load(f)
            
    def filter_data(self, item_ids=None):
        valid_data = []
        item_id_set = set(item_ids) if item_ids is not None else None
        # Loop through the metadata and check for file existence
        for _, row in self.metadata.iterrows():
            music_id =  str(row['music_id'])
            file_path = os.path.join(self.audio_dir, row['filename'])
            
            if os.path.exists(file_path) and str(music_id) in self.embeddings:
                embedding_data = self.embeddings[music_id]
                
                if isinstance(embedding_data[0], list):
                    embeddings = torch.tensor(embedding_data)  # [num_embeddings, embedding_dim]
                else:
                    embeddings = torch.tensor([embedding_data]) # [1, embedding_dim]
                    
                if item_id_set is None or int(music_id) in item_id_set:  # filter by item_ids
                    valid_data.append({'music_id': music_id,
                                    'file_path': file_path,
                                    'embedding': embeddings})
        self.data = valid_data
        print(f"Loaded {len(self.data)} audio samples.")
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['music_id'], item['file_path'], item['embedding']

    def __len__(self):
        return len(self.data)