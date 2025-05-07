# =============================================================================
# File: Attributes_processor.py
# Description: Defines a processors for audio features.
# Author: Vojtěch Nekl
# Created: 24.03.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================


import os
import torch
import pandas as pd
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
import json

class Attributes_Processor:
    """
    A processor for handling and preprocessing audio attribute features.

    Supports operations such as standardization, scaling by occurrence,
    encoding categorical variables, and feature selection based on configuration.
    """

    def __init__(self, prepr_config, train_ids):              
        self.scale_AST_att = prepr_config["scale_AST_att"]
        self.standardize_ess_att = prepr_config["standardize_ess_att"]
        self.standardize_ast_att = prepr_config["standardize_ast_att"]
        self.which_attributes = prepr_config["which_attributes"]
        self.train_ids = train_ids
        
        csv_path = prepr_config["csv_path"]
        AST_occurence_path = prepr_config["AST_occurence_path"]
        
        if not os.path.exists(csv_path):
            print(f"Path {csv_path} does not exist.")
            raise FileNotFoundError
        else:
            df_all = pd.read_csv(csv_path)
            print(f"Loaded {len(df_all)} rows from {csv_path}")
        
        if not os.path.exists(AST_occurence_path):
            if self.standardize_ess_att is not None or self.scale_AST_att is not None:
                print(f"Path {AST_occurence_path} does not exist.")
                raise FileNotFoundError
        else:
            with open(AST_occurence_path, 'r') as f:
                self.occurrences = json.load(f)
                
        self.process_attributes(df_all)
                
    def get_input_dim(self):
        """Returns the number of dimensions of a single data point."""
        sample = self.df.iloc[0, 1:].to_numpy()  # Exclude 'music_id' column if it's the first column
        return sample.shape[0]

    def process_attributes(self, df_all):
        # Convert 'music_id' to string type
        df_all['music_id'] = df_all['music_id'].astype(str)
        
        # Store DataFrame without filtering by IDs
        self.df = df_all.copy()

        # Normalize Essentia columns if required
        if self.standardize_ess_att or self.standardize_ast_att:
            self.standardize()
            
        # Scale AST features with the inverse of their occurrence if required
        if self.scale_AST_att:
            self.scale_AST()
        
        # Encode musical key and scale
        self.encodeScale()
        self.encode_musical_key()
        
        # Filter features
        self.filter_features()

    def filter_features(self):
        """Filters features based on prepr_config["which_attributes"] setting."""
        if self.which_attributes == "ast":
            self.df = self.df[[col for col in self.df.columns if col in self.occurrences or col == "music_id"]]
        elif self.which_attributes == "library":
            self.df = self.df[[col for col in self.df.columns if col not in self.occurrences or col == "music_id"]]


    def standardize(self):
        columns_to_standardize = []
        if self.standardize_ess_att:
            for column in self.df.columns:
                if column not in self.occurrences and column != "music_id":
                    columns_to_standardize.append(column)
                    
        if self.standardize_ast_att:
            for column in self.df.columns:
                if column in self.occurrences:
                    columns_to_standardize.append(column)
        
        print(f"Standardizing {len(columns_to_standardize)} attributes")
 
        train_df = self.df[self.df["music_id"].isin(self.train_ids)]
        
        for column in columns_to_standardize:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                # found on training set only
                col_mean = train_df[column].mean()
                col_std = train_df[column].std()
                if col_std > 0:
                    self.df[column] = (self.df[column] - col_mean) / col_std
                else:
                    print(f"Column {column} has the same max and min value. Values will be setted to 0.")
                    self.df[column] = 0
                
    def scale_AST(self):
        sum_occ = sum(self.occurrences.values())
        for column in self.df.columns:
            if column in self.occurrences:
                col_occ = self.occurrences[column]
                self.df[column] = self.df[column] * (sum_occ/col_occ)
    
    def encodeScale(self):
        if 'Musical Scale' in self.df.columns:
            self.df['is_major'] = self.df['Musical Scale'].apply(lambda x: 1 if x == 'major' else 0)
            self.df = self.df.drop(columns=['Musical Scale']).copy()
        
    def circular_encoding(self, key):
        """Encodes a musical key (0-11) using sine-cosine encoding."""
        theta = (2 * np.pi / 12) * key
        return np.cos(theta), np.sin(theta)
        
    def encode_musical_key(self):
        key_mapping = {
            "C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, 
            "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11
        }
    
        self.df[["key_cos", "key_sin"]] = self.df["Musical Key"].map(key_mapping).apply(self.circular_encoding).apply(pd.Series)
        self.df.drop(columns=["Musical Key"], inplace=True)
         
    def read_audios_into_dict(self, all_ids):
        filtered_df = self.df.set_index("music_id").loc[all_ids]
        return filtered_df.apply(lambda row: {"input_values": torch.tensor(row.tolist())}, axis=1).to_dict()
    
    def read_audios_from_dict(self, ids, dictionary):
        t = []
        for id in tqdm(ids):
            t.append(dictionary[id])
        tt = {}
        for key in tqdm(t[0].keys()):
            if isinstance(t[0][key], list):
                tt[key] = [j for i in [x[key] for x in t] for j in i]
            else:
                tt[key] = torch.vstack([x[key] for x in t])
        return tt

    def get_ordered_tensor(self, ids):
        """
        Returns a numpy matrix with attributes ordered by the provided list of ids.
        """
        df_indexed = self.df.set_index("music_id")

        missing_ids = [i for i in ids if i not in df_indexed.index]
        if missing_ids:
            raise ValueError(f"Missing attributes for the following ids: {missing_ids}")

        return np.vstack([df_indexed.loc[i].values for i in ids])

class Embedding_Processor:
    """
    A processor for loading and optionally standardizing audio embeddings.
    """

    def __init__(self, prepr_config, train_ids):
        path = prepr_config["embeddings_json_path"]
        standardize = prepr_config["standardize"]
        self.train_ids = train_ids
        
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            raise FileNotFoundError
        else:
            with open(path, 'r') as f:
                embeddings = json.load(f)
                print (f"Loaded {len(embeddings)} embeddings from {path}")
        
        if standardize:
            self.standardize(embeddings)                
        else:
            self.embeddings = embeddings
    
    def get_input_dim(self):
        """Returns the number of dimensions of a single data point."""
        sample_id = next(iter(self.embeddings))  # Get the first key
        return len(self.embeddings[sample_id])
    
    def standardize(self, embeddings):
        # standardize whole dataset based on training set
        train_embeddings = {k: v for k, v in embeddings.items() if k in self.train_ids}
        train_embeddings = np.vstack(list(train_embeddings.values()))
        mean = train_embeddings.mean(axis=0)
        std = train_embeddings.std(axis=0) + 1e-8 # avoid division by zero
        self.embeddings = {k: (v - mean) / std for k, v in embeddings.items()}
        
                
    def read_audios_into_dict(self, all_ids):
        # adds input_values as key
        dict_att = {id: {"input_values": torch.tensor(self.embeddings[id], dtype=torch.float32)} for id in all_ids if id in self.embeddings}
        
        # Check if all IDs were found in the embeddings
        missing_ids = set(all_ids) - set(dict_att.keys())
        if missing_ids:
            raise ValueError(f"The following IDs were not found in the embeddings: {missing_ids}")
            
        return dict_att
    
    def read_audios_from_dict(self, ids, dictionary):
        t = []
        for id in tqdm(ids):
            t.append(dictionary[id])
        tt = {}
        for key in tqdm(t[0].keys()):
            if isinstance(t[0][key], list):
                tt[key] = [j for i in [x[key] for x in t] for j in i]
            else:
                tt[key] = torch.vstack([x[key] for x in t])
        return tt
    
    def get_ordered_tensor(self, ids):
        """
        Returns a numpy matrix with embeddings ordered by the provided list of ids.
        """
        missing_ids = [i for i in ids if i not in self.embeddings]
        if missing_ids:
            raise ValueError(f"Missing embeddings for the following ids: {missing_ids}")
        
        return np.vstack([self.embeddings[i] for i in ids])