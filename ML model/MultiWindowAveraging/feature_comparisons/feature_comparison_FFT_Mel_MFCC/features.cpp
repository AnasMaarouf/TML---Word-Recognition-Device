#include "features.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "eml_fft.h"
#include "eml_audio.h"
#include "eml_vector.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr int FFT_N = 4096;
static constexpr float EPS = 1e-12f;

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
        std::fprintf(stderr,
            "magnitude_spectrum_cpp: out_len mismatch (got %d, expected %d)\n",
            out_len, expected_out_len);
        return;
    }

    std::vector<float> real(FFT_N, 0.0f);
    std::vector<float> imag(FFT_N, 0.0f);
    std::vector<float> fft_sin(FFT_N / 2, 0.0f);
    std::vector<float> fft_cos(FFT_N / 2, 0.0f);

    const int copy_len = std::min(n, FFT_N);
    for (int i = 0; i < copy_len; ++i) {
        const float w = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (FFT_N - 1));
        real[i] = x[i] * w;
    }

    EmlFFT fft;
    fft.length = FFT_N / 2;
    fft.sin = fft_sin.data();
    fft.cos = fft_cos.data();

    EmlError err = eml_fft_fill(fft, FFT_N);
    if (err != EmlOk) {
        std::fprintf(stderr, "eml_fft_fill failed: %d\n", (int)err);
        return;
    }

    err = eml_fft_forward(fft, real.data(), imag.data(), FFT_N);
    if (err != EmlOk) {
        std::fprintf(stderr, "eml_fft_forward failed: %d\n", (int)err);
        return;
    }

    for (int k = 0; k < out_len; ++k) {
        freqs[k] = static_cast<float>(k) * fs / static_cast<float>(FFT_N);
        mag[k] = std::sqrt(real[k] * real[k] + imag[k] * imag[k]);
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
    int idx = 1; // skip DC
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

bool mel_spectrum_cpp(const float* x, int n, float fs, float* mel_out, int n_mels) {
    if (n_mels <= 0) {
        std::fprintf(stderr, "mel_spectrum_cpp: n_mels must be > 0\n");
        return false;
    }

    std::vector<float> audio_buf(FFT_N, 0.0f);
    std::vector<float> temp_buf(FFT_N, 0.0f);
    std::vector<float> fft_sin(FFT_N / 2, 0.0f);
    std::vector<float> fft_cos(FFT_N / 2, 0.0f);

    const int copy_len = std::min(n, FFT_N);
    for (int i = 0; i < copy_len; ++i) {
        audio_buf[i] = x[i];
    }

    EmlFFT fft;
    fft.length = FFT_N / 2;
    fft.sin = fft_sin.data();
    fft.cos = fft_cos.data();

    EmlError err = eml_fft_fill(fft, FFT_N);
    if (err != EmlOk) {
        std::fprintf(stderr, "mel_spectrum_cpp: eml_fft_fill failed: %d\n", (int)err);
        return false;
    }

    EmlAudioMel mel_params;
    mel_params.n_mels = n_mels;
    mel_params.fmin = 0.0f;
    mel_params.fmax = fs / 2.0f;
    mel_params.n_fft = FFT_N;
    mel_params.samplerate = static_cast<int>(fs);

    EmlVector inout;
    inout.data = audio_buf.data();
    inout.length = FFT_N;

    EmlVector temp;
    temp.data = temp_buf.data();
    temp.length = FFT_N;

    err = eml_audio_melspectrogram(mel_params, fft, inout, temp);
    if (err != EmlOk) {
        std::fprintf(stderr, "mel_spectrum_cpp: eml_audio_melspectrogram failed: %d\n", (int)err);
        return false;
    }

    // output ligger i de første n_mels samples af inout ifølge eml_audio.h
    for (int i = 0; i < n_mels; ++i) {
        mel_out[i] = audio_buf[i];
    }

    return true;
}

bool mfcc_features_cpp(const float* x, int n, float fs, float* mfcc_out, int n_mfcc, int n_mels) {
    if (n_mfcc <= 0 || n_mels <= 0) {
        std::fprintf(stderr, "mfcc_features_cpp: invalid dimensions\n");
        return false;
    }

    std::vector<float> mel_vals(n_mels, 0.0f);
    if (!mel_spectrum_cpp(x, n, fs, mel_vals.data(), n_mels)) {
        return false;
    }

    // log-mel
    for (int i = 0; i < n_mels; ++i) {
        mel_vals[i] = std::log(mel_vals[i] + EPS);
    }

    // DCT-II
    for (int k = 0; k < n_mfcc; ++k) {
        double sum = 0.0;
        for (int m = 0; m < n_mels; ++m) {
            sum += mel_vals[m] * std::cos((double)M_PI * k * (m + 0.5) / n_mels);
        }
        mfcc_out[k] = static_cast<float>(sum);
    }

    return true;
}