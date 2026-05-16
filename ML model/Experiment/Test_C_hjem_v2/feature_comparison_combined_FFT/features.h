#pragma once

static constexpr int FEATURE_FFT_N = 4096;
static constexpr int SIMPLE_FEATURES_N = 9;
static constexpr int N_MFCC = 13;
static constexpr int N_MELS = 20;
static constexpr int COMBINED_FEATURES_N = SIMPLE_FEATURES_N + N_MFCC + N_MELS;

float rms_energy_cpp(const float* x, int n);
float zero_crossing_rate_cpp(const float* x, int n);

bool magnitude_spectrum_fft_cpp(
    const float* x,
    int n,
    float fs,
    float* freqs,
    float* mag,
    int out_len
);

float spectral_centroid_cpp(const float* freqs, const float* mag, int n);
float spectral_rolloff_cpp(const float* freqs, const float* mag, int n, float roll_percent);
float dominant_frequency_cpp(const float* freqs, const float* mag, int n);
float band_energy_cpp(const float* freqs, const float* mag, int n, float f_low, float f_high);

bool mel_features_emlearn_cpp(
    const float* x,
    int n,
    float fs,
    float* mel_out,
    int n_mels
);

bool mfcc_features_emlearn_cpp(
    const float* x,
    int n,
    float fs,
    float* mfcc_out,
    int n_mfcc
);

bool extract_combined_features_cpp(
    const float* x,
    int n,
    float fs,
    float* features_out
);