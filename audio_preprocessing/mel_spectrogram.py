import numpy as np

class MelSpectrogram:
    def __init__(self, num_mels=80, sample_rate=16000, n_fft=512, f_min=0, f_max=8000):
        """
        Args:
            num_mels:    number of mel filterbank channels
            sample_rate: audio sample rate (must match FFT)
            n_fft:       FFT size (must match FastFourierTransformation.n_fft)
            f_min:       minimum frequency for mel filterbank
            f_max:       maximum frequency for mel filterbank
        """
        self.num_mels    = num_mels
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.f_min       = f_min
        self.f_max       = f_max

        self._filterbank = self._build_mel_filterbank()

    def _build_mel_filterbank(self):
        m_min = 2595 * np.log10(1 + self.f_min / 700)
        m_max = 2595 * np.log10(1 + self.f_max / 700)

        mel_points = np.linspace(m_min, m_max, self.num_mels + 2)
        hz_points  = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        bin_points = np.clip(bin_points, 0, self.n_fft // 2) 

        n_freqs = self.n_fft // 2 + 1
        filters = np.zeros((self.num_mels, n_freqs), dtype=np.float32)

        for i in range(1, self.num_mels + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]

            if center > left:
                for k in range(left, center):
                    filters[i-1, k] = (k - left) / (center - left)
            if right > center:
                for k in range(center, right):
                    filters[i-1, k] = (right - k) / (right - center)

            area = right - left
            if area > 0:
                filters[i-1] *= 2.0 / area
        return filters 

    def calculate_filterbank(self):
        return self._filterbank

    def apply_filterbank(self, filterbank, power_spectrogram):
        return power_spectrogram @ filterbank.T

    def apply_log(self, mel_energies):
        return np.log(mel_energies + 1e-9)

    def build_mel_spectrogram(self, fft_magnitudes):
        """
        Build log-mel spectrogram from framed FFT magnitudes.

        Args:
            fft_magnitudes: 2D array of shape (num_frames, n_fft//2+1)
                            — output of FastFourierTransformation.apply_fft()

        Returns:
            log_mel: 2D array of shape (num_frames, num_mels)
                     e.g. (300, 80) for ~3 seconds of audio
        """
        if fft_magnitudes.ndim == 1:
            fft_magnitudes = fft_magnitudes[np.newaxis, :]

        power_spectrogram = fft_magnitudes ** 2        
        mel_energies = self.apply_filterbank(self._filterbank,
                                                   power_spectrogram)
        log_mel = self.apply_log(mel_energies)        
        return log_mel 