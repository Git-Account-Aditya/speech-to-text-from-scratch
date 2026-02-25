import numpy as np 

class MelSpectrogram:
    def __init__(self, num_mels=80, sample_rate=16000, n_fft=2048, f_min=0, f_max=8000):
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max

    def _build_mel_filterbank(self):
        # Convert freq range to mel
        m_min = 2595 * np.log10(1 + self.f_min / 700)
        m_max = 2595 * np.log10(1 + self.f_max / 700)

        # N+2 equally spaced mel points → convert back to Hz → map to FFT bins
        mel_points = np.linspace(m_min, m_max, self.num_mels + 2)
        hz_points  = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Build filterbank matrix: shape [n_filters, n_fft//2 + 1]
        n_freqs = self.n_fft // 2 + 1
        filters = np.zeros((self.num_mels, n_freqs))

        for i in range(1, self.num_mels + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]

            # Rising slope (guard against division by zero when left == center)
            if center > left:
                for k in range(left, center):
                    filters[i-1, k] = (k - left) / (center - left)
            # Falling slope (guard against division by zero when center == right)
            if right > center:
                for k in range(center, right):
                    filters[i-1, k] = (right - k) / (right - center)

            # Area normalization — ensures narrow and wide filters contribute equal energy
            area = right - left
            if area > 0:
                filters[i-1] *= 2.0 / area

        return filters

    def calculate_filterbank(self):
        return self._build_mel_filterbank()

    def apply_filterbank(self, filterbank, spectrogram):
        return spectrogram @ filterbank.T
    
    def apply_log(self, mel_energies):
        return np.log(mel_energies + 1e-9)

    def build_mel_spectrogram(self, fft_magnitudes):
        """
        Build mel spectrogram from pre-computed FFT magnitudes.

        Args:
            fft_magnitudes: FFT magnitude output from FastFourierTransformation.apply_fft()
                            - 1D array of shape (n_fft//2 + 1,) for a single frame
                            - 2D array of shape (num_frames, n_fft//2 + 1) for multiple frames

        Returns:
            log_mel_energies: shape (num_mels,) for 1D input, or (num_frames, num_mels) for 2D
        """
        # Convert magnitudes to power spectrogram
        power_spectrogram = fft_magnitudes ** 2

        # If 1D (single frame), reshape to 2D for matrix multiply
        squeeze = False
        if power_spectrogram.ndim == 1:
            power_spectrogram = power_spectrogram[np.newaxis, :]
            squeeze = True

        filterbank = self.calculate_filterbank()
        mel_energies = self.apply_filterbank(filterbank, power_spectrogram)
        log_mel_energies = self.apply_log(mel_energies)

        # Return same dimensionality as input
        if squeeze:
            log_mel_energies = log_mel_energies.squeeze(0)

        return log_mel_energies
