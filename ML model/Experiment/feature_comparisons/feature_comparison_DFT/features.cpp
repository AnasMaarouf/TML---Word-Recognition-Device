#include "features.h"
#include <cmath>
#include <vector>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
        bool prev = x[i - 1] < 0.0f;
        bool curr = x[i] < 0.0f;
        if (prev != curr) crossings++;
    }
    return static_cast<float>(crossings) / static_cast<float>(n - 1);
}

// Simpel DFT til validering på laptop.
// Langsom, men fin til at validere mod Python.
void magnitude_spectrum_cpp(const float* x, int n, float fs, float* freqs, float* mag, int out_len) {
    std::vector<float> xw(n);

    for (int i = 0; i < n; ++i) {
        float w = 0.5f - 0.5f * std::cos(2.0 * M_PI * i / (n - 1));
        xw[i] = x[i] * w;
    }

    for (int k = 0; k < out_len; ++k) {
        double real = 0.0;
        double imag = 0.0;

        for (int t = 0; t < n; ++t) {
            double angle = -2.0 * M_PI * k * t / n;
            real += xw[t] * std::cos(angle);
            imag += xw[t] * std::sin(angle);
        }

        freqs[k] = static_cast<float>(k) * fs / n;
        mag[k] = std::sqrt(real * real + imag * imag);
    }
}

float spectral_centroid_cpp(const float* freqs, const float* mag, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += freqs[i] * mag[i];
        den += mag[i];
    }
    return static_cast<float>(num / (den + 1e-12));
}

float spectral_rolloff_cpp(const float* freqs, const float* mag, int n, float roll_percent) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += mag[i];

    double threshold = roll_percent * total;
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
        if (mag[i] > mag[idx]) idx = i;
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