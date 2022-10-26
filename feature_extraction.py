import numpy as np
import scipy.signal


def butter_bandpass(lowcut, highcut, fs, order=5):
  return scipy.signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = scipy.signal.lfilter(b, a, data)
  return y

def get_alpha_band(data, freq):
  return butter_bandpass_filter(data, 8, 12, freq)
  # return butter_bandpass_filter(data, 8, 16, freq)

def get_beta_band(data, freq):
  return butter_bandpass_filter(data, 12, 30, freq)
  # return butter_bandpass_filter(data, 16, 32, freq)

def get_gamma_band(data, freq):
  return butter_bandpass_filter(data, 30, 100, freq)
  # return butter_bandpass_filter(data, 32, 96, freq)


class FeatureExtraction:
  def __init__(self, data, freq):
    self.data = data
    self.freq = freq
  
  def line_length(self):
    return np.sum(np.abs(np.diff(self.data)))

  def spectral_alpha_power(self):
    return np.sum(np.power(get_alpha_band(self.data, self.freq), 2))
  
  def spectral_beta_power(self):
    return np.sum(np.power(get_beta_band(self.data, self.freq), 2))
  
  def spectral_gamma_power(self):
    return np.sum(np.power(get_gamma_band(self.data, self.freq), 2))
  
  def get_features(self):
    return [
      self.line_length(),
      self.spectral_alpha_power(),
      self.spectral_beta_power(),
      self.spectral_gamma_power(),
    ]
