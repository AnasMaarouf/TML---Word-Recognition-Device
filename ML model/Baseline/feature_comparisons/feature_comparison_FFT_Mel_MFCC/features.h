#pragma once

float rms_energy_cpp(const float* x, int n);
float zero_crossing_rate_cpp(const float* x, int n);

// output arrays skal have længde FFT_N/2 + 1
void magnitude_spectrum_cpp(const float* x, int n, float fs, float* freqs, float* mag, int out_len);

float spectral_centroid_cpp(const float* freqs, const float* mag, int n);
float spectral_rolloff_cpp(const float* freqs, const float* mag, int n, float roll_percent);
float dominant_frequency_cpp(const float* freqs, const float* mag, int n);
float band_energy_cpp(const float* freqs, const float* mag, int n, float f_low, float f_high);

// Mel + MFCC
bool mel_spectrum_cpp(const float* x, int n, float fs, float* mel_out, int n_mels);
bool mfcc_features_cpp(const float* x, int n, float fs, float* mfcc_out, int n_mfcc, int n_mels);