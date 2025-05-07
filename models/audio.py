# =============================================================================
# File: audio.py
# Description: Defines a wrapper for the audio-beeFormer model.
# Author: Vojtěch Nekl
# Created: 14.11.2024
# Notes: Created as part of the bachelor's thesis work. Inspiration is driven from image-beeFormer.
# =============================================================================


from config import config
from utils import *

import transformers
from transformers import AutoModel, ASTModel, AutoModelForAudioClassification, AutoProcessor
import pandas as pd
import math 
import os

from Audio_processor.AST_processor import AST_Processor

def read_audio(id, fn, path, look_up_table, device):
    if id in look_up_table:
        audio = look_up_table[id]
    else:
        print(f"Audio with id {id} not found in look_up_table.")
        return None
    
    audio_path = os.path.join(path, audio['filename'])
    
    try:
        return fn(audio_path)
    except:
        print(f"Tokenization error for {audio_path}. Replacing with empty waveform.")
        empty_spectograms = torch.zeros(1, 1, 1024, 128).to(device)
        return {"input_values": empty_spectograms}


def generate_lookup_table(path, metadata_path):
    metadata = pd.read_csv(metadata_path)
    look_up_table = {}
    # Loop through the metadata and check for file existence
    for _, row in metadata.iterrows():
        file_path = os.path.join(path, row['filename'])
        if os.path.exists(file_path):
            look_up_table[str(row['music_id'])] = {'music_id': row['music_id'], 'filename': row['filename']} 

    return look_up_table

def read_audios_into_dict(ids, fn, path, metadata_path ="_datasets/audiodataset/recombee-audio-files-samples.csv", device = "cuda"):
    t = {}
    look_up_table = generate_lookup_table(path, metadata_path)
    for id in tqdm(ids, desc="Preprocessing audios"):
        t[id] = read_audio(id, fn, path, look_up_table, device)
    return t

def read_audios_from_dict(ids, dictionary):
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


class AudioModel(torch.nn.Module):
    """
    Defines a wrapper for audio models, allowing for tokenization, encoding, and pooling over audio spectrograms.

    Args:
        model_name (str): Name of the pre-trained model to load.
        device (str): Device to run the model on ('cpu' or 'cuda').
        prepr_config (dict): Configuration for preprocessing.
        pooling (str, optional): Pooling strategy for sequence outputs ('CLS' or 'mean'). Defaults to 'CLS'.
        num_spectograms (int, optional): Number of spectrograms per sample. Defaults to 1.
        more_spec_pooling (str, optional): Strategy for pooling over spectrograms ('mean' or 'max'). Defaults to 'mean'.
        trust_remote_code (bool, optional): Flag for trusting remote code when loading models. Defaults to False.
    """
    def __init__(self, model_name, device, prepr_config, pooling="CLS", num_spectograms=1,
                 more_spec_pooling = "mean", trust_remote_code=False):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.prepr_config = prepr_config
        self.pooling = pooling.lower()
        assert self.pooling in ("cls", "mean")
        self.num_spectograms = num_spectograms
        self.more_spec_pooling = more_spec_pooling
        assert self.more_spec_pooling in ("mean", "max")
        
        self._load(model_name, trust_remote_code, device)
        self.to(device)

    def tokenize(self, file_name):
        try:
            return self.processor(file_name)
        except Exception as e:
            print(f"Exception during tokenization: {e}")
            raise
    
    
    def forward(self, data):
        all_spectrograms = []
        spectrograms_per_sample = []

        for spectrograms in data['input_values']:
            spectrograms_per_sample.append(len(spectrograms))
            all_spectrograms.extend(spectrograms)

        # Stack into one large batch: shape [total_spectrograms, spec_x, spec_y]
        all_spectrograms_tensor = torch.stack(all_spectrograms).to(self.device)

        # Inference: pass entire batch of spectrograms at once
        if self.model_name == "MIT/ast-finetuned-audioset-10-10-0.4593":
            out = self.model(input_values=all_spectrograms_tensor).last_hidden_state
        elif self.model_name == "bookbot/distil-ast-audioset":
            out = self.model.audio_spectrogram_transformer(input_values=all_spectrograms_tensor).last_hidden_state

        # Pooling over sequence dim
        if self.pooling == "cls":
            pooled = out[:, 0]
        else:
            pooled = out.mean(dim=1)

        # Now reconstruct original batch from pooled chunks
        embeddings = []
        idx = 0
        for num_chunks in spectrograms_per_sample:
            sample_embeddings = pooled[idx:idx + num_chunks]
            if self.more_spec_pooling == "mean":
                aggregated = sample_embeddings.mean(dim=0)
            else:
                aggregated = sample_embeddings.max(dim=0)[0]
            embeddings.append(aggregated)
            idx += num_chunks

        final_embeddings = torch.stack(embeddings)
        return {'sentence_embedding': final_embeddings}



    def move_tokens_to_device(self, tokens, ind_min=None, ind_max=None):
        if ind_min is not None and ind_max is not None:
            return {k: v[ind_min:ind_max].to(self.device) if isinstance(v, torch.Tensor) else v[ind_min:ind_max] for k, v in tokens.items()}
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

    def encode(self, tokenized_waveforms, batch_size=32, show_progress_bar=True):
        l = get_first_item(tokenized_waveforms).shape[0]
        max_i = math.ceil(l / batch_size)
        ret = []
        with torch.no_grad():
            for i in tqdm(range(max_i), desc="Encoding for ELSA", disable=not show_progress_bar):
                ind = i * batch_size
                ind_min = ind
                ind_max = ind + batch_size
                tokens_to_encode = self.move_tokens_to_device(tokenized_waveforms, ind_min, ind_max)
                ret.append(self(tokens_to_encode)['sentence_embedding'])
            return torch.vstack(ret)

    def save(self, model_name=None):
        save_to = self.model_name if model_name is None else model_name
        print(f"Saving model to: {save_to}")
        self.model.save_pretrained(save_to)
        # all configuration details are stored in setup.csv
        
        #self.processor.save_pretrained(self.model_name if model_name is None else model_name)

    def _load(self, model_name, trust_remote_code, device):
        # here, I would specify different processors for different models
        if model_name == "MIT/ast-finetuned-audioset-10-10-0.4593" or model_name == "bookbot/distil-ast-audioset":
            self.processor = AST_Processor(device, self.prepr_config, self.num_spectograms)
        
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        #if isinstance(self.model, ASTModel):
        #    print("Creating AST model")
        #    self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
       
        if model_name == "MIT/ast-finetuned-audioset-10-10-0.4593":
            print("Creating AST model")
            self.model = ASTModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        elif model_name == "bookbot/distil-ast-audioset":
            print("Creating distilled AST model")
            self.model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        else:
            # Fallback to AutoModel if the model name does not match the specific cases
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)