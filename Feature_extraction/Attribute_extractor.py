# =============================================================================
# File: Attribute_extractor.py
# Description: This script defines classes for extracting audio features.
# Author: Vojtěch Nekl
# Created: 28.04.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import ASTModel, AutoModelForAudioClassification, ASTForAudioClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json 
import csv

from AST_ontology_mask import *
from dataset import *
from Attribute_extractor_config import *
from essentia.standard import MusicExtractor
import essentia

essentia.log.infoActive = False

import sys

print(os.getcwd())
sys.path.append(os.getcwd())
ffmpeg_path = './ffmpeg-7.0.2-amd64-static/ffmpeg'
ffprobe_path = './ffmpeg-7.0.2-amd64-static/ffprobe'

os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
os.environ["PATH"] += os.pathsep + os.path.dirname(ffprobe_path)


class EmbeddingGenerator:
    """
    Extracts audio embeddings from a dataset of audio files using a pre-trained model.
    """
    def __init__(self, audio_path, metadata_csv, device, audio_conf,
                 model_name, normalize, global_mean, global_std,
                 spectograms_per_embedding, pooling, batch_size,
                 output_directory, item_ids=None, ):
        self.audio_path = audio_path
        self.metadata_csv = metadata_csv
        self.device = device
        self.audio_conf = audio_conf
        self.model_name = model_name
        self.normalize = normalize
        self.global_mean = global_mean
        self.global_std = global_std
        self.spectograms_per_embedding = spectograms_per_embedding
        self.pooling = pooling
        self.batch_size = batch_size
        self.output_directory = output_directory
        self.item_ids = item_ids

        print("Loading model and dataloader")
        self.model, self.dataloader = self._load_model_and_dataloader()
        
    def _load_model_and_dataloader(self):
        dataset = AudioDataset(self.audio_path, self.metadata_csv, self.device, self.audio_conf, self.spectograms_per_embedding, self.normalize, self.global_mean, self.global_std, self.item_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        if self.model_name == "MIT/ast-finetuned-audioset-10-10-0.4593":
            print("Creating AST model")
            model = ASTModel.from_pretrained(self.model_name).to(self.device)
            model = torch.nn.DataParallel(model)
        elif self.model_name == "bookbot/distil-ast-audioset":
            print("Creating distilled AST model")
            model = AutoModelForAudioClassification.from_pretrained(self.model_name).to(self.device)
        
        return model, dataloader

    def generate_embeddings(self):
        print("Computing embeddings")
        all_embeddings = {}
        
        for batch in tqdm(self.dataloader, desc="Processing batches"):
            # Extract the stack of spectrograms and corresponding music IDs from the batch
            fbanks_stacks, music_ids = batch
            fbanks_stacks = fbanks_stacks.to(self.device)

            with torch.no_grad():
                # Process each stack of spectrograms
                embeddings_per_song = []
                for fbanks in fbanks_stacks:
                    if self.model_name == "MIT/ast-finetuned-audioset-10-10-0.4593":
                        outputs = self.model(fbanks)
                    elif self.model_name == "bookbot/distil-ast-audioset":
                        outputs = self.model.audio_spectrogram_transformer(fbanks)
                    
                    embedding = outputs.pooler_output  # [6, embedding_dim]
                    
                    # Apply max or mean pooling across the stack dimension to combine 10s-embeddings
                    if self.pooling == "mean":
                        final_embedding = torch.mean(embedding, dim=0).cpu().numpy()
                    elif self.pooling == "max":
                        final_embedding = torch.max(embedding, dim=0)[0].cpu().numpy()
                    else:
                        final_embedding = embedding.cpu().numpy() # letting all num_spectograms embeddings in the list
                        
                    embeddings_per_song.append(final_embedding)

                for music_id, final_embedding in zip(music_ids, embeddings_per_song):
                    all_embeddings[music_id.item()] = final_embedding.tolist()

        return self.save_embeddings_to_file(all_embeddings)
        
    
    def save_embeddings_to_file(self, all_embeddings):
        distilled = "_distilled" if self.model_name == "bookbot/distil-ast-audioset" else ""
        pooling_desc = f"{self.pooling}" if self.pooling else "nopooling"
        
        file_name = f"{len(all_embeddings)}_embeddings_{self.spectograms_per_embedding}0s_{pooling_desc}{distilled}.json"
        path = os.path.join(self.output_directory, file_name)
        with open(path, 'w') as f:
            json.dump(all_embeddings, f, indent=4)

        print(f"Stored embeddings for {len(all_embeddings)} songs as {file_name} into ")  
        return path          
            
            
class AudioAttributeExtractor:
    def __init__(self, dataset_conf, model_conf, device):
        self.audio_conf = dataset_conf["audio_conf"]
        self.ontology_path = dataset_conf["ontology_path"]
        self.model_name = dataset_conf["model_name"]
        self.device = device

        self.thresholds = model_conf["thresholds"]
        self.masks = model_conf["masks"]
        self.is_music = model_conf["is_music"]
        self.output_directory = model_conf["output_directory"]
        self.output_file = model_conf["output_file"]
        self.print = model_conf["print"]
        
        self.results = {}
        self.classification_layer = ASTForAudioClassification.from_pretrained(dataset_conf["model_name"]).to(self.device)
        self.config = self.classification_layer.config
        self.labels = self.config.id2label
        self.masks, self.mask_all = self.create_masks()
    
    def save_to_results_dictionary(self, music_id, output):
        self.results[music_id] = {}
        for j, label in self.labels.items():
            if self.mask_all[j]:
                self.results[music_id][label] = output[j].tolist()

    def save_batch_to_csv(self, is_first_batch=False):
        # Collect all possible feature names
        all_keys = set()
        for features in self.results.values():
            all_keys.update(features.keys())  # Collect all unique feature names
        
        all_keys = list(all_keys)
        # Open file in append mode ('a') if appending; write mode ('w') for first batch
        mode = 'w' if is_first_batch else 'a'
        
        path_to_open = os.path.join(self.output_directory, self.output_file)
        with open(path_to_open, mode, newline='') as f:
            writer = csv.writer(f)
            
            # Write header only if this is the first batch
            if is_first_batch:
                writer.writerow(["music_id"] + all_keys)
                self.first_batch = False
            
            # Write each song's features as a row
            for music_id, features in self.results.items():
                row = [music_id]  # Start with the music ID
                for key in all_keys:
                    row.append(features.get(key, "N/A"))  # Default missing values
                writer.writerow(row)

        self.results.clear()
    
    def forward(self, music_ids, file_paths, embeddings, is_first_batch=False):
        embeddings = embeddings.to(self.device)
        
        # Flatten embeddings for classification if multiple exist per file
        batch_size, *embedding_dims = embeddings.shape  # (batch_size, num_embeddings, embedding_dim) or (batch_size, embedding_dim)
        
        if len(embedding_dims) == 2:  # Multiple embeddings per music_id
            num_embeddings = embedding_dims[0]
            embeddings = embeddings.view(batch_size * num_embeddings, embedding_dims[1])  # Flatten: [batch_size * num_embeddings, embedding_dim]

        with torch.no_grad():
            outputs = self.classification_layer.classifier(embeddings)  # Classify each embedding separately
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(batch_size, num_embeddings, -1) if len(embedding_dims) == 2 else outputs  # Reshape back if needed
            
            # Now, apply mean pooling across embeddings per file
            if len(embedding_dims) == 2:
                outputs = outputs.mean(dim=1)  # Pooling over embeddings -> [batch_size, num_classes]
                
            outputs = outputs.cpu().numpy()  
    
        for i in range(outputs.shape[0]):
            output = outputs[i]
            self.save_to_results_dictionary(music_ids[i], output)
            if self.print: 
                print(f'\nItem {music_ids[i]}, {file_paths[i]}')
                self._apply_masks(output)
            
            self.extract_essentia_features(file_paths[i], self.audio_conf["sample_rate"], music_ids[i])
        
        self.save_batch_to_csv(is_first_batch)
    
    def extract_essentia_features(self, filename, sample_rate, music_id):
        """
        Extract high-level features using Essentia's MusicExtractor.
        """
    
        music_extractor = MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                         rhythmStats=['mean', 'stdev'],
                                         tonalStats=['mean', 'stdev'])
        
        features, _ = music_extractor(filename)
        
        def get_feature(features, key, default='Not available'):
            try:     
                return features[key]
            except KeyError:
                return default
        
        self.results[music_id]['Average Loudness'] = get_feature(features, 'lowlevel.average_loudness')  # Overall perceived loudness
        self.results[music_id]['Dynamic Complexity'] = get_feature(features, 'lowlevel.dynamic_complexity')  # Variation in loudness
        self.results[music_id]['Loudness Integrated'] = get_feature(features, 'lowlevel.loudness_ebu128.integrated')  # Integrated loudness over track
        self.results[music_id]['Loudness Range'] = get_feature(features, 'lowlevel.loudness_ebu128.loudness_range')  # Loudness variability
        
        self.results[music_id]['BPM_essentia'] = get_feature(features, 'rhythm.bpm')  # Beats per minute (tempo)
        self.results[music_id]['Beats Count'] = get_feature(features, 'rhythm.beats_count')  # Total number of beats detected
        self.results[music_id]['Beats Loudness'] = get_feature(features, 'rhythm.beats_loudness.mean')  # Average loudness of beats
        self.results[music_id]['Danceability'] = get_feature(features, 'rhythm.danceability')  # Suitability for dancing
        self.results[music_id]['Onset Rate'] = get_feature(features, 'rhythm.onset_rate')  # Percussiveness (rate of new sound onsets)

        self.results[music_id]['Spectral Energy'] = get_feature(features, 'lowlevel.spectral_energy.mean')  # Overall signal energy (loudness/power)
        self.results[music_id]['Spectral Centroid'] = get_feature(features, 'lowlevel.spectral_centroid.mean')  # Brightness of sound (higher = more treble)
        self.results[music_id]['Spectral Entropy'] = get_feature(features, 'lowlevel.spectral_entropy.mean')  # Complexity/noisiness of sound
        self.results[music_id]['Spectral Rolloff'] = get_feature(features, 'lowlevel.spectral_rolloff.mean')  # Edge frequency where most energy is contained
        self.results[music_id]['Spectral Flux'] = get_feature(features, 'lowlevel.spectral_flux.mean')  # Rate of spectral changes

        self.results[music_id]['Chords Strength'] = get_feature(features, 'tonal.chords_strength.mean')  # Strength of harmonic content
        self.results[music_id]['Chords Changes Rate'] = get_feature(features, 'tonal.chords_changes_rate')  # Frequency of chord changes
        self.results[music_id]['Musical Key'] = get_feature(features, 'tonal.key_edma.key')  # Estimated musical key
        self.results[music_id]['Musical Scale'] = get_feature(features, 'tonal.key_edma.scale')  # Major or minor scale

        self.results[music_id]['Pitch Salience'] = get_feature(features, 'lowlevel.pitch_salience.mean')  # How dominant the pitch is in the sound
        self.results[music_id]['Track Length'] = get_feature(features, 'metadata.audio_properties.length')  # Track duration
    
        # Extract Chroma Features (HPCP - Harmonic Pitch Class Profile)
        hpcp = get_feature(features, 'tonal.hpcp.mean', [0]*12)  # HPCP is essentially Chroma
        pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i, pitch in enumerate(pitch_classes):
            self.results[music_id][f'Chroma_essentia_{pitch}'] = float(hpcp[i]) if i < len(hpcp) else 0
        
        if self.print:
            self.print_high_level_features()


    def create_masks(self):
        mask_generator = OntologyMaskGenerator(self.ontology_path, self.model_name)
        masks = mask_generator.generate_masks(self.masks)
        if self.is_music:
            masks["music"] = np.zeros(len(self.labels), dtype=int)
            masks["music"][137] = 1
            self.thresholds.append(0.00)
        
        mask_aggregated = np.zeros(len(self.labels), dtype=int)
        for mask in masks.values():
            mask_aggregated += mask

        return masks, mask_aggregated

    def _apply_masks(self, output):
        """ Only print the top 5 predictions for each mask """
        
        for j, (category, mask) in enumerate(self.masks.items()):
            if self.print:
                print(f'Mask: {category} ,   threshold: {self.thresholds[j]}')
            
            masked_output = output * mask
            # filtering out everything under threshold ( or 0.00 )
            #np.where(masked_output >= self.thresholds[j], masked_output, 0)
            np.where(masked_output >= 0, masked_output, 0)
            # Sort and print top 5 predictions
            sorted_indexes = np.argsort(masked_output)[::-1][:5]  # Get top 5 indices

            # filtering out everything under threshold
            sorted_indexes = [idx for idx in sorted_indexes if masked_output[idx] > 0][:5]

            for k in range(len(sorted_indexes)):
                index = sorted_indexes[k]
                if index in self.labels:  
                    sound_name = self.labels[index]
                    probability = masked_output[index]
                    above_threshold = probability >= self.thresholds[j]
                    if self.print:
                        mark = '*' if above_threshold else ''
                        print(f'-  {mark} {sound_name}: {probability:.4f}')
                else:
                    print(f'Warning: index {index} not found in labels.')
                    
    def print_high_level_features(self):
        """
        Print extracted high-level features.
        """
        print(f"Essentia Alternative Features:")
        
        print(f"  Average Loudness: {self.results['Average Loudness']}")
        print(f"  Dynamic Complexity: {self.results['Dynamic Complexity']}")
        print(f"  Loudness Integrated: {self.results['Loudness Integrated']}")
        print(f"  Loudness Range: {self.results['Loudness Range']}")
        print(f"  BPM: {self.results['BPM_essentia']}")
        print(f"  Beats Count: {self.results['Beats Count']}")
        print(f"  Beats Loudness: {self.results['Beats Loudness']}")
        print(f"  Danceability: {self.results['Danceability']}")
        print(f"  Onset Rate: {self.results['Onset Rate']}")
        print(f"  Spectral Energy: {self.results['Spectral Energy']}")
        print(f"  Spectral Centroid: {self.results['Spectral Centroid']}")
        print(f"  Spectral Entropy: {self.results['Spectral Entropy']}")
        print(f"  Spectral Rolloff: {self.results['Spectral Rolloff']}")
        print(f"  Spectral Flux: {self.results['Spectral Flux']}")
        print(f"  Chords Strength: {self.results['Chords Strength']}")
        print(f"  Chords Changes Rate: {self.results['Chords Changes Rate']}")
        print(f"  Musical Key: {self.results['Musical Key']} {self.results['Musical Scale']}")
        print(f"  Pitch Salience: {self.results['Pitch Salience']}")
        print(f"  Track Length: {self.results['Track Length']}")
        print(f"  Sample Rate: {self.results['Sample Rate']}")
        
        pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i in range(12):
            print(f"  Chroma Essentia {pitch_classes[i]}: {self.results[f'Chroma_essentia_{pitch_classes[i]}']}")
        print()