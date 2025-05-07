# =============================================================================
# File: audio_att.py
# Description: Defines a wrapper for the features-beaFormer model.
# Author: Vojtěch Nekl
# Created: 11.1.2025
# Notes: Created as part of the bachelor's thesis work.
# =============================================================================


from utils import *

import torch.nn as nn
import torch.nn.functional as F
import math 

from Audio_processor.Attributes_processor import Attributes_Processor, Embedding_Processor

class DynamicNN(nn.Module):
    """
    A dynamic fully connected neural network with specified number of hidden layers.
    
    Args:
        input_dim (int): Input feature dimension.
        num_hidden_layers (int): Number of hidden layers.
        hidden_dimension (int): The dimension (number of units) in each hidden layer.
        output_dimension (int): Output feature dimension.
    """
    def __init__(self, input_dim, num_hidden_layers, hidden_dimension, output_dimension):
        super(DynamicNN, self).__init__()
        layers = []
        prev_dim = input_dim

        # Dynamically create hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dimension))
            layers.append(nn.ReLU())
            prev_dim = hidden_dimension
        
        layers.append(nn.Linear(prev_dim, output_dimension))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class AudioAtt_wrapper(nn.Module):
    """
    Wrapper class for feature-beaFormer model, integrates different attribute or embedding processors.
    
    Args:
        device (str): Device to run the model on ('cpu' or 'cuda').
        model_type (str): The type of model ('audio_att_model' or 'audio_emb_model').
        prepr_config (dict): Configuration for data preprocessing.
        train_ids (list): List of training data identifiers.
        num_hidden_layers (int): Number of hidden layers for the model.
        hidden_dimension (int): Number of neurons in each hidden layer.
        output_dimension (int): Dimension of the output embeddings.
    """
    def __init__(self, device, model_type, prepr_config, train_ids, num_hidden_layers, hidden_dimension, output_dimension):
        super().__init__()
        self.device = device
        self.model_type = model_type
        
        # initialize the processor
        if model_type == "audio_att_model":
            self.processor = Attributes_Processor(prepr_config, train_ids)
        elif model_type == "audio_emb_model":
            self.processor = Embedding_Processor(prepr_config, train_ids)
        
        input_dim = self.processor.get_input_dim()
        
        self.model = DynamicNN(input_dim, num_hidden_layers, hidden_dimension, output_dimension)
        self.to(device)

    def tokenize(self, data):
        return self.processor(data)
    
    def forward(self, data):
        input_values = data['input_values']
        final_embeddings = self.model(input_values)
        return {'sentence_embedding': final_embeddings}

    def move_tokens_to_device(self, tokens, ind_min=None, ind_max=None):
        if ind_min is not None and ind_max is not None:
            return {k: v[ind_min:ind_max].to(self.device) if isinstance(v, torch.Tensor) else v[ind_min:ind_max] for k, v in tokens.items()}
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

    def encode(self, tokenized_waveforms, batch_size=32, show_progress_bar=False):
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
        final_name = model_name if model_name else self.specific_model + self.model_type + ".pt"
        
        if final_name is None or not isinstance(final_name, str):
            raise ValueError("Error: Model name is None or not a string!")

        print(f"Saving model to: {final_name}")
        torch.save(self.model.state_dict(), final_name)
        print("Model saved successfully.")