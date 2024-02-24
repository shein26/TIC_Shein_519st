import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt

n = 500
Fs = 1000
F_max = 3
noise_amplitude = 10
random_signal = np.random.normal(0, noise_amplitude, n)
time = np.arange(n) / Fs
cutoff_frequency = F_max / (Fs / 2)
sos = signal.butter(3, cutoff_frequency, 'low', output='sos')
filtered_signal = signal.sosfiltfilt(sos, random_signal)
plt.figure(figsize=(10, 6))
plt.plot(time, filtered_signal, linewidth=1, color='blue')
plt.xlabel('Час (секунди)', fontsize=12)
plt.ylabel('Амплітуда сигналу', fontsize=12)
plt.title('Спектр сигналу з максимальною частотою F_max=3 Гц', fontsize=14)
plt.grid(True)
plt.savefig('./figures/filtered_signal.png', dpi=300)
spectrum = fft.fft(filtered_signal)
spectrum = np.abs(fft.fftshift(spectrum))
freqs = fft.fftfreq(n, 1 / Fs)
freqs = fft.fftshift(freqs)
plt.figure(figsize=(10, 6))
plt.plot(freqs, spectrum, linewidth=1, color='red')
plt.xlabel('Частота (Гц)', fontsize=12)
plt.ylabel('Амплітуда спектру', fontsize=12)
plt.title('Спектр сигналу з максимальною частотою F_max=3 Гц', fontsize=14)
plt.grid(True)
plt.savefig('./figures/spectrum_of_filtered_signal.png', dpi=300)
plt.show()
