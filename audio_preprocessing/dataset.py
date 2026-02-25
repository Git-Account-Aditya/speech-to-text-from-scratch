import numpy as np 
import torch 
import soundfile as sf
import io


class SpeechDataset:
    def __init__(
        self,
        speech_dataset,
        num_examples=None,
        tokenizer=None
    ):
        self.dataset = speech_dataset
        self.tokenizer = tokenizer
        self.num_examples = min(num_examples, len(speech_dataset)) if num_examples is not None else len(speech_dataset)
    
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]

        # Decode audio bytes from parquet into numpy array
        audio_data = item['audio']
        if isinstance(audio_data, dict) and 'bytes' in audio_data:
            audio_bytes = audio_data['bytes']
        elif isinstance(audio_data, (bytes, bytearray)):
            audio_bytes = audio_data
        else:
            raise ValueError(f"Unexpected audio format: {type(audio_data)}")

        samples, _ = sf.read(io.BytesIO(audio_bytes))
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        waveform = torch.from_numpy(samples).float()
        text = item['transcription'].upper()

        if self.tokenizer:
            encoded_txt = self.tokenizer.encode(text)
            return {"audio": waveform, "text": text, "input_ids": encoded_txt.ids}
        
        return {"audio": waveform, "text": text}