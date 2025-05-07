# =============================================================================
# File: AST_processor.py
# Description: Defines a custom dataset class for audio preprocessing for AST.
# Author: Vojtěch Nekl
# Created: 15.02.2025
# Notes: Created as part of the bachelor's thesis work. Inspiration is driven by the original AST preprocessor.
# =============================================================================


import torchaudio
import torch
import os
import pandas as pd
import torchaudio.transforms as T
from pydub import AudioSegment

class AST_Processor:
    """
    Custom processor for audio files to generate spectrograms suitable for AST model input.

    Attributes:
        device (torch.device): Device for computation (e.g., 'cpu' or 'cuda').
        prepr_config (dict): Preprocessing configuration including sample rate, normalization parameters, etc.
        num_spectograms (int): Number of spectrogram segments to extract from each input audio.
    """
    def __init__(self, device, prepr_config, num_spectograms):
                
        self.device = device
        self.num_spectograms = num_spectograms
        self.sample_rate = prepr_config["sample_rate"]
        self.normalize = prepr_config["normalize"]
        self.norm_mean = prepr_config["mean"]
        self.norm_std = prepr_config["std"]
        self.target_length = prepr_config["target_length"]
        self.num_mel_bins = prepr_config["num_mel_bins"]
    
    def _load_resample(self, filename):
        waveform, sample_rate = torchaudio.load(filename)
        waveform = waveform #.to(self.device)

        resample_rate = self.sample_rate
        if sample_rate != resample_rate:
            resampler = T.Resample(sample_rate, resample_rate) # .to(self.device)
            waveform = resampler(waveform)
        
        return waveform, resample_rate

    def _convert_to_wav(self, filename):
        # Set the output .wav file path
        wav_filename = filename.replace('.m4a', '.wav')
        if not os.path.exists(wav_filename):  # Convert only if .wav file doesn't exist
            audio = AudioSegment.from_file(filename, format="m4a")
            audio.export(wav_filename, format="wav")
        return wav_filename
    
    def __call__(self, file_name):
        processed_waveform = self._process_waveform(file_name)
        return {"input_values": processed_waveform.unsqueeze(0)}  # Return the tensor with an added batch dimension
    
    def _process_waveform(self, file_name):
        wav_name = self._convert_to_wav(file_name)
        waveform, sr = self._load_resample(wav_name)
        waveform = waveform # .to(self.device)
        
        # converting stereo audio to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10
        )# .to(self.device)

        target_length = self.target_length
        n_frames = fbank.shape[0]

        # Add configuration for specific number of spectograms, like one should be taken from middle...
        spectograms=[]
        for i in range(self.num_spectograms):
            start_idx = i * target_length
            end_idx = start_idx + target_length

            # pad if necessary
            if end_idx > n_frames:
                segment = fbank[start_idx:n_frames]
                segment = torch.nn.functional.pad(segment, (0, 0, 0, target_length - segment.shape[0]))
            else:
                segment = fbank[start_idx:end_idx]

            if self.normalize:
                segment = self.normalize_fbank(segment)
                
            spectograms.append(segment)
        return torch.stack(spectograms)
    
    def normalize_fbank(self, fbank):
        if  self.norm_mean is not None and self.norm_std is not None:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        else:
            print("Mean or Std is None")
            print("Continuing without normalization")
            
        return fbank