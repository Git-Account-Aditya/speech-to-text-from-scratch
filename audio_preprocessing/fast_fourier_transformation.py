import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt


class FastFourierTransformation:
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=400):
        """
        Args:
            sample_rate:  target sample rate (Hz)
            n_fft:        FFT window size in samples (512 = 32ms at 16kHz)
            hop_length:   stride between frames in samples (160 = 10ms at 16kHz)
            win_length:   analysis window length in samples (400 = 25ms at 16kHz)
        """
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.win_length  = win_length
        self.window = np.hanning(win_length)

    def apply_fft(self, audio_samples):
        """
        Apply framed FFT to audio samples using a sliding window.

        Each frame is:
          1. Extracted with hop_length stride
          2. Multiplied by a Hann window
          3. Zero-padded to n_fft if win_length < n_fft
          4. Passed through rfft

        Args:
            audio_samples: 1D numpy array of audio samples

        Returns:
            magnitudes: 2D array of shape (num_frames, n_fft//2 + 1)
                        — one row per time frame, ready for mel filterbank
        """
        audio_samples = audio_samples.astype(np.float32)

        pad = self.win_length // 2
        audio_samples = np.pad(audio_samples, (pad, pad), mode='reflect')

        num_frames = 1 + (len(audio_samples) - self.win_length) // self.hop_length
        magnitudes  = np.zeros((num_frames, self.n_fft // 2 + 1), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio_samples[start : start + self.win_length]
            frame = frame * self.window

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            magnitudes[i] = np.abs(np.fft.rfft(frame, n=self.n_fft))
        return magnitudes 

    def load_audio(self, audio_path):
        """
        Load a WAV file and return audio samples + sample rate.

        Args:
            audio_path: path to the .wav file

        Returns:
            samples:     1D numpy array of audio samples
            sample_rate: sample rate in Hz
        """
        samples, sample_rate = sf.read(audio_path)

        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        return samples, sample_rate


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir  = os.path.normpath(os.path.join(script_dir, '..', 'training_data', 'audio'))

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files in {audio_dir}\n")

    fft = FastFourierTransformation()

    for audio_file in audio_files[:3]:
        audio_path = os.path.join(audio_dir, audio_file)
        samples, sample_rate = fft.load_audio(audio_path)

        print(f"File: {audio_file}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration:    {len(samples) / sample_rate:.2f}s")
        print(f"  Samples:     {samples.shape}")

        magnitudes = fft.apply_fft(samples)
        print(f"  FFT output:  {magnitudes.shape}  ← (num_frames, n_fft//2+1)")
        print(f"  Frames:      {magnitudes.shape[0]}")
        print()

    # Plot last file
    plt.figure(figsize=(10, 4))
    plt.imshow(magnitudes.T, aspect='auto', origin='lower')
    plt.title(f'FFT Magnitudes over time — {audio_file}')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')
    plt.colorbar()
    plt.tight_layout()
    plt.show()