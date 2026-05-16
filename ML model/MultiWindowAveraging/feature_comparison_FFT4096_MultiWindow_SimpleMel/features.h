#pragma once

static constexpr int FEATURE_FFT_N = 4096;
static constexpr int FEATURE_FFT_HOP = FEATURE_FFT_N / 2;

static constexpr int SIMPLE_FEATURES_N = 9;
static constexpr int N_MELS = 20;
static constexpr int SIMPLE_MEL_FEATURES_N = SIMPLE_FEATURES_N + N_MELS;

float rms_energy_cpp(const float* x, int n);
float zero_crossing_rate_cpp(const float* x, int n);

bool extract_simple_mel_features_cpp(
    const float* x,
    int n,
    float fs,
    float* features_out
);