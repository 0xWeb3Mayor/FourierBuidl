from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    from scipy.fft import fft, fftfreq, ifft
except ImportError:
    from numpy.fft import fft, fftfreq, ifft


@dataclass(frozen=True)
class FFTFeatures:
    dominant_freq: float
    cycle_days: float | None
    amplitude: float
    noise_ratio: float
    deviation_score: float
    n_samples: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "dominant_freq": self.dominant_freq,
            "cycle_days": self.cycle_days,
            "amplitude": self.amplitude,
            "noise_ratio": self.noise_ratio,
            "deviation_score": self.deviation_score,
            "n_samples": self.n_samples,
        }


def _prepare_prices(prices: list[float]) -> np.ndarray:
    if len(prices) < 2:
        raise ValueError("FFT analysis requires at least 2 price points")
    prices_array = np.asarray(prices, dtype=float)
    if not np.all(np.isfinite(prices_array)):
        raise ValueError("Price series contains non-finite values")
    return np.clip(prices_array, 0.0, 1.0)


def _detrend(prices_array: np.ndarray) -> np.ndarray:
    n = len(prices_array)
    return prices_array - np.linspace(prices_array[0], prices_array[-1], n)


def reconstruct_dominant_cycle(prices: list[float], dominant_freq: float | None = None) -> np.ndarray:
    prices_array = _prepare_prices(prices)
    n = len(prices_array)
    prices_detrended = _detrend(prices_array)
    window = np.hanning(n)
    spectrum = fft(prices_detrended * window)
    freqs = fftfreq(n, d=1)

    if dominant_freq is None:
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_amplitudes = np.abs(spectrum[pos_mask])
        if len(pos_amplitudes) == 0:
            return np.zeros(n)
        dominant_freq = float(pos_freqs[int(np.argmax(pos_amplitudes))])

    matches = np.where(np.isclose(freqs, dominant_freq))[0]
    if len(matches) == 0:
        return np.zeros(n)

    dominant_idx = int(matches[0])
    reconstruction_spectrum = np.zeros(n, dtype=complex)
    reconstruction_spectrum[dominant_idx] = spectrum[dominant_idx]
    reconstruction_spectrum[(-dominant_idx) % n] = spectrum[(-dominant_idx) % n]
    return np.real(ifft(reconstruction_spectrum))


def run_fft_analysis(prices: list[float]) -> dict[str, float | int | None]:
    """
    Takes hourly price series and returns frequency-domain features.
    Prices are clipped to prediction-market-native [0, 1] bounds.
    """
    prices_array = _prepare_prices(prices)
    n = len(prices_array)

    prices_detrended = _detrend(prices_array)
    window = np.hanning(n)
    prices_windowed = prices_detrended * window

    spectrum = fft(prices_windowed)
    freqs = fftfreq(n, d=1)

    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_amplitudes = np.abs(spectrum[pos_mask])

    if len(pos_amplitudes) == 0 or float(np.sum(pos_amplitudes**2)) == 0.0:
        return FFTFeatures(
            dominant_freq=0.0,
            cycle_days=None,
            amplitude=0.0,
            noise_ratio=1.0,
            deviation_score=0.0,
            n_samples=n,
        ).as_dict()

    dominant_idx = int(np.argmax(pos_amplitudes))
    dominant_freq = float(pos_freqs[dominant_idx])
    dominant_amplitude = float(pos_amplitudes[dominant_idx])
    cycle_days = float((1 / dominant_freq) / 24) if dominant_freq > 0 else None

    low_freq_cutoff = 1 / (24 * 3)
    high_freq_power = float(np.sum(pos_amplitudes[pos_freqs > low_freq_cutoff] ** 2))
    total_power = float(np.sum(pos_amplitudes**2))
    noise_ratio = high_freq_power / total_power if total_power > 0 else 1.0

    dominant_component = reconstruct_dominant_cycle(prices, dominant_freq)
    current_deviation = float(abs(prices_detrended[-1] - dominant_component[-1]))
    deviation_score = (
        current_deviation / dominant_amplitude if dominant_amplitude > 0 else 0.0
    )

    return FFTFeatures(
        dominant_freq=dominant_freq,
        cycle_days=cycle_days,
        amplitude=dominant_amplitude,
        noise_ratio=float(noise_ratio),
        deviation_score=float(deviation_score),
        n_samples=n,
    ).as_dict()
