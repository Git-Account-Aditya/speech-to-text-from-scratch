import numpy as np
import torch 

from dataset import SpeechDataset
from fast_fourier_transformation import FastFourierTransformation
from mel_spectrogram import MelSpectrogram
from custom_tokenizer import create_tokenizer
from huggingface_hub import hf_hub_download
import pandas as pd

if __name__ == "__main__":
    try:
        # Download the parquet file directly — completely bypasses torchcodec/FFmpeg
        print("Downloading dataset...")
        parquet_path = hf_hub_download(
            repo_id="m-aliabbas/idrak_timit_subsample1",
            filename="data/train-00000-of-00001-aeb35d2d506d38bf.parquet",
            repo_type="dataset"
        )

        # Read with pandas
        dataset = pd.read_parquet(parquet_path)

        print(f"Dataset columns: {list(dataset.columns)}")
        print(f"Dataset shape: {dataset.shape}")
        print(type(dataset))
        # Create tokenizer
        tokenizer = create_tokenizer()

        # Create dataloader
        speech_dataset = SpeechDataset(dataset, tokenizer=tokenizer)
        print(speech_dataset[0])

        def collate_fn(batch):
            # Pad audio to same length
            audios = [item['audio'] for item in batch]
            audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

            texts = [item['text'] for item in batch]

            # Pad input_ids to same length
            input_ids = [torch.tensor(item['input_ids']) for item in batch]
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)

            return {'audio': audios_padded, 'text': texts, 'input_ids': input_ids_padded}

        dataloader = torch.utils.data.DataLoader(speech_dataset, batch_size=1, collate_fn=collate_fn)
        
        for batch in dataloader:
            print(batch['audio'].shape)
            print(batch['text'])
            print(batch['input_ids'].shape)
            break

        # Create MelSpectrogram instance
        fft = FastFourierTransformation()
        mel_spectrogram = MelSpectrogram()

        # Get the first batch
        batch = next(iter(dataloader))
        audio = batch['audio']  # Shape: (batch_size, num_samples)

        # Compute FFT (per sample — expects 1D numpy array)
        audio_numpy = audio[0].numpy()  # First sample in batch
        fft_output = fft.apply_fft(audio_numpy)
        print(f"FFT Output Shape: {fft_output.shape}")  # (num_freq_bins,)

        # n_fft must match the FFT output: rfft returns n//2+1 bins, so n_fft = (bins-1)*2
        n_fft = (fft_output.shape[0] - 1) * 2
        mel_spectrogram = MelSpectrogram(n_fft=n_fft)

        # Compute Mel Spectrogram from FFT magnitudes
        mel_output = mel_spectrogram.build_mel_spectrogram(fft_output)
        print(f"Mel Spectrogram Shape: {mel_output.shape}")  # (num_mels,)
    
    except Exception as e:
        print(f"Error: {e}")