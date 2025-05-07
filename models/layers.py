# =============================================================================
# File: layers.py
# Original Author: Vojtěch Vančura
# Description: Defines custom Keras layers for the Beeformer model.
# Notes: Imported as part of the project without any modifications.
# =============================================================================

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import sentence_transformers
import torch

from keras.layers import TorchModuleWrapper
#from images import ImageModel

from audio import AudioModel
from audio_att import AudioAtt_wrapper

# basic elsa model as a keras layer (usebale at other keras models)
class LayerELSA(keras.layers.Layer):
    def __init__(self, n_dims, n_items, device):
        super(LayerELSA, self).__init__()
        self.device = device
        self.A = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_dims, n_items])))

    def parameters(self, recurse=True):
        return [self.A]

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self):
        self.to(self.device)
        sample_input = torch.ones([self.A.shape[0]]).to(self.device)
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x):
        A = torch.nn.functional.normalize(self.A, dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return keras.activations.relu(xAAT - x)


# keras wrapper around sentence transformers object
class LayerSBERT(keras.layers.Layer):
    def __init__(self, model, device, tokenized_sentences):
        super(LayerSBERT, self).__init__()
        self.device = device
        self.sbert = TorchModuleWrapper(model.to(device))
        self.tokenize_ = self.sb().tokenize
        self.tokenized_sentences = tokenized_sentences
        self.build()

    def sb(self):
        for module in self.sbert.modules():
            #------------------------------------------------------------------------------------------------------------------------------
            if isinstance(module, sentence_transformers.SentenceTransformer) or isinstance(module, ImageModel) or isinstance(module, AudioModel) or isinstance(module, AudioAtt_wrapper):
                return module

    def parameters(self, recurse=True):
        return self.sbert.parameters()

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def tokenize(self, inp):
        # move tokenized tensors to device and return tokenized sentences
        return {k: v.to(self.device) for k, v in self.tokenize_(inp).items()}

    def build(self):
        self.to(self.device)
        sample_input = {k: v[:2].to(self.device) for k, v in self.tokenized_sentences.items()}
        _ = self.call(sample_input)
        self.track_module_parameters()

    def freeze_parameters(self, num_layers_to_freeze):
        for name, param in self.sbert.named_parameters():
            for i in range(num_layers_to_freeze):
                layer_prefix = f"model.audio_spectrogram_transformer.encoder.layer.{i}"
                if name.startswith(layer_prefix):
                    param.requires_grad = False
                    print(f"[frozen] {name}")

                
    def call(self, x):
        # just call sentence transformer model
        return self.sbert.forward(x)["sentence_embedding"]