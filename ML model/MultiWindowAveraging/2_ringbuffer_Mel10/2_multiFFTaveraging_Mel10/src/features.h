#pragma once

static constexpr int FEATURE_FFT_N = 4096;
static constexpr int FEATURE_FFT_HOP = FEATURE_FFT_N / 2;

static constexpr int N_MELS = 10;
static constexpr int MEL_FEATURES_N = N_MELS;

bool extract_mel_features_cpp(
    const float* x,
    int n,
    float fs,
    float* features_out
);

#include <stdint.h>

bool extract_mel_features_i16_cpp(
    const int16_t* x,
    int n,
    float fs,
    float* features_out
);