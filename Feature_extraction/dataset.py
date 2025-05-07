# =============================================================================
# File: dataset.py
# Description: Defines a custom dataset class for loading audio files and converting them to Mel spectrograms.
# Author: Vojtěch Nekl
# Created: 23.12.2024
# Notes: Created as part of the bachelor's thesis work. Inspiration is driven by the original AST preprocessor.
# =============================================================================

import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from pydub import AudioSegment
import torchaudio.transforms as T

#from config import device

class AudioDataset(Dataset):
    def __init__(self, audio_dir, metadata_csv, device, audio_conf, num_spect = 1 ,normalize = False, mean = None, std = None, item_ids=None):
        """
        Custom Dataset for loading audio recordings.
        :param audio_dir: Directory containing audio files.
        :param metadata_csv: CSV file containing Audio IDs and names.
        :param audio_conf: Dictionary with audio loading and preprocessing settings.
        :param num_spect: Number of spectograms with 1024 timeframes per file.
        :param normalize: Flag to return normalized spectograms.
        :mean: Mean of data for normalization.
        :std: Standard deviation of data for normalization
        :param item_ids: List of item IDs to load. If None, all items will be loaded
        """
        self.audio_dir = audio_dir
        self.metadata = pd.read_csv(metadata_csv)
        self.audio_conf = audio_conf
        self.num_spect = num_spect
        self.device = device
        self.sampling_rate = audio_conf["sample_rate"]
        self.filter_data(item_ids)
        
        
        self.normalize = normalize
        self.norm_mean = mean
        self.norm_std = std

    def filter_data(self, item_ids=None):
        valid_data = []
        item_id_set = set(item_ids) if item_ids is not None else None
        # Loop through the metadata and check for file existence
        for _, row in self.metadata.iterrows():
            file_path = os.path.join(self.audio_dir, row['filename'])
            if os.path.exists(file_path):
                if item_id_set is None or int(row['music_id']) in item_id_set:
                    valid_data.append({'music_id': row['music_id'], 'filename': row['filename']})

        self.data = valid_data
        print(f"Loaded {len(self.data)} audio samples.")

    def _convert_to_wav(self, filename):
        # Set the output .wav file path
        wav_filename = filename.replace('.m4a', '.wav')
        if not os.path.exists(wav_filename):  # Convert only if .wav file doesn't exist
            audio = AudioSegment.from_file(filename, format="m4a")
            audio.export(wav_filename, format="wav")
        return wav_filename

    
    def _wav2fbank(self, filename):
        # Load audio file
        wav_filename = self._convert_to_wav(filename)
        waveform, sr = torchaudio.load(wav_filename)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        resample_rate = self.sampling_rate
        if sr != resample_rate:
            resampler = T.Resample(sr, resample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform - waveform.mean()  # Centering the waveform
        
        # Convert the waveform to a Mel spectrogram (fbank)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=resample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.audio_conf.get('num_mel_bins', 128),
            dither=0.0,
            frame_shift=10
        )
        
        target_length = self.audio_conf.get('target_length', 1024)
        n_frames = fbank.shape[0]


        spectograms = []
        for i in range(self.num_spect):
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

    def __getitem__(self, index):
        """ Datum _format 
        datum = {
            'music_id': 'track_001',
            'filename': 'track_001.wav'
        }
        """
        
        datum = self.data[index]
        file_path = os.path.join(self.audio_dir, datum['filename'])
        music_id = datum['music_id']
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return "file not found"
    
        fbank = self._wav2fbank(file_path)
   
        return fbank, music_id

    def __len__(self):
        return len(self.data)