#include "Particle.h"
#include "features.h"

#include <cmath>
#include <algorithm>

#include "eml_fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr int FFT_N = 4096;

// Store buffers statisk/globalt
static float g_real[FFT_N];
static float g_imag[FFT_N];
static float g_fft_sin[FFT_N / 2];
static float g_fft_cos[FFT_N / 2];

float rms_energy_cpp(const float* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum / n + 1e-12);
}

float zero_crossing_rate_cpp(const float* x, int n) {
    int crossings = 0;
    for (int i = 1; i < n; ++i) {
        const bool prev = x[i - 1] < 0.0f;
        const bool curr = x[i] < 0.0f;
        if (prev != curr) {
            crossings++;
        }
    }
    return static_cast<float>(crossings) / static_cast<float>(n - 1);
}

void magnitude_spectrum_cpp(const float* x, int n, float fs, float* freqs, float* mag, int out_len) {
    const int expected_out_len = FFT_N / 2 + 1;
    if (out_len != expected_out_len) {
        Serial.printf("magnitude_spectrum_cpp: out_len mismatch (got %d, expected %d)\r\n",
            out_len, expected_out_len);
        return;
    }

    for (int i = 0; i < FFT_N; ++i) {
        g_real[i] = 0.0f;
        g_imag[i] = 0.0f;
    }
    for (int i = 0; i < FFT_N / 2; ++i) {
        g_fft_sin[i] = 0.0f;
        g_fft_cos[i] = 0.0f;
    }

    const int copy_len = std::min(n, FFT_N);
    for (int i = 0; i < copy_len; ++i) {
        const float w = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (FFT_N - 1));
        g_real[i] = x[i] * w;
    }

    EmlFFT fft;
    fft.length = FFT_N / 2;
    fft.sin = g_fft_sin;
    fft.cos = g_fft_cos;

    EmlError err = eml_fft_fill(fft, FFT_N);
    if (err != EmlOk) {
        Serial.printf("eml_fft_fill failed: %d\r\n", (int)err);
        return;
    }

    err = eml_fft_forward(fft, g_real, g_imag, FFT_N);
    if (err != EmlOk) {
        Serial.printf("eml_fft_forward failed: %d\r\n", (int)err);
        return;
    }

    for (int k = 0; k < out_len; ++k) {
        freqs[k] = static_cast<float>(k) * fs / static_cast<float>(FFT_N);
        mag[k] = std::sqrt(g_real[k] * g_real[k] + g_imag[k] * g_imag[k]);
    }
}

float spectral_centroid_cpp(const float* freqs, const float* mag, int n) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += freqs[i] * mag[i];
        den += mag[i];
    }
    return static_cast<float>(num / (den + 1e-12));
}

float spectral_rolloff_cpp(const float* freqs, const float* mag, int n, float roll_percent) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        total += mag[i];
    }

    const double threshold = roll_percent * total;
    double cumsum = 0.0;

    for (int i = 0; i < n; ++i) {
        cumsum += mag[i];
        if (cumsum >= threshold) {
            return freqs[i];
        }
    }
    return freqs[n - 1];
}

float dominant_frequency_cpp(const float* freqs, const float* mag, int n) {
    int idx = 1;
    for (int i = 2; i < n; ++i) {
        if (mag[i] > mag[idx]) {
            idx = i;
        }
    }
    return freqs[idx];
}

float band_energy_cpp(const float* freqs, const float* mag, int n, float f_low, float f_high) {
    double power_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (freqs[i] >= f_low && freqs[i] < f_high) {
            power_sum += mag[i] * mag[i];
        }
    }
    return static_cast<float>(power_sum);
}