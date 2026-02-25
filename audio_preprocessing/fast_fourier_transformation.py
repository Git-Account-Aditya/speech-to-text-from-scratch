import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt 


class FastFourierTransformation:
    def __init__(self):
        pass

    def apply_fft(self, audio_samples):
        """
        Apply FFT to audio samples and return the frequency magnitudes.
        
        Args:
            audio_samples: numpy array of audio samples (1D)
            
        Returns:
            frequencies: array of frequency bins (positive only)
            magnitudes: array of magnitude values for each frequency
        """
        n = len(audio_samples)
        fft_result = np.fft.rfft(audio_samples)
        magnitudes = np.abs(fft_result)
        return magnitudes

    def load_audio(self, audio_path):
        """
        Load a WAV file and return audio samples + sample rate.
        
        Args:
            audio_path: path to the .wav file
            
        Returns:
            samples: numpy array of audio samples
            sample_rate: sample rate in Hz
        """
        samples, sample_rate = sf.read(audio_path)

        # If stereo, convert to mono by averaging channels
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        return samples, sample_rate


if __name__ == "__main__":
    # Use path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(script_dir, '..', 'training_data', 'audio')
    audio_dir = os.path.normpath(audio_dir)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files in {audio_dir}\n")

    fft = FastFourierTransformation()

    for audio_file in audio_files[:3]:
        audio_path = os.path.join(audio_dir, audio_file)

        # Load audio properly as numpy samples
        samples, sample_rate = fft.load_audio(audio_path)
        print(f"File: {audio_file}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(samples) / sample_rate:.2f}s")
        print(f"  Samples shape: {samples.shape}")
        print(f"  Sample values: {samples}")

        # Apply FFT
        magnitudes = fft.apply_fft(samples)
        print(f"  FFT magnitudes shape: {magnitudes.shape}")
        print(f"  Peak magnitude: {np.max(magnitudes):.2f}")
        print(f"  Mean magnitude: {np.mean(magnitudes):.2f}")
        print()

    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(samples)
    plt.title(f'FFT Magnitudes for {audio_file}')
    plt.xlabel('Frequency Bins')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()