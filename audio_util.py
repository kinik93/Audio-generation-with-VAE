import librosa
import numpy as np


def inv_magphase(mag, phase_angle):

    return (np.cos(phase_angle) + 1j*np.sin(phase_angle))*mag


def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):
  """Iterative algorithm for phase retrival from a magnitude spectrogram.

  Args:
    mag: Magnitude spectrogram.
    phase_angle: Initial condition for phase.
    n_fft: Size of the FFT.
    hop: Stride of FFT. Defaults to n_fft/2.
    num_iters: Griffin-Lim iterations to perform.

  Returns:
    audio: 1-D array of float32 sound samples.
  """
  fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
  ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
  complex_specgram = inv_magphase(mag, phase_angle)
  for i in range(num_iters):
    audio = librosa.istft(complex_specgram, **ifft_config)
    if i != num_iters - 1:
      complex_specgram = librosa.stft(audio, **fft_config)
      _, phase = librosa.magphase(complex_specgram)
      phase_angle = np.angle(phase)
      complex_specgram = inv_magphase(mag, phase_angle)
  return audio