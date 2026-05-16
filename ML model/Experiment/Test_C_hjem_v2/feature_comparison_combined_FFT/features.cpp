#include "features.h"

#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "eml_fft.h"
#include "eml_audio.h"
#include "eml_vector.h"

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

        if (prev != curr) {
            crossings++;
        }
    }

    return static_cast<float>(crossings) / static_cast<float>(n - 1);
}

bool magnitude_spectrum_fft_cpp(
    const float* x,
    int n,
    float fs,
    float* freqs,
    float* mag,
    int out_len
) {
    static float real[FEATURE_FFT_N];
    static float imag[FEATURE_FFT_N];
    static float fft_sin[FEATURE_FFT_N / 2];
    static float fft_cos[FEATURE_FFT_N / 2];

    for (int i = 0; i < FEATURE_FFT_N; ++i) {
        real[i] = 0.0f;
        imag[i] = 0.0f;
    }

    int copy_len = std::min(n, FEATURE_FFT_N);

    for (int i = 0; i < copy_len; ++i) {
        float w = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (FEATURE_FFT_N - 1));
        real[i] = x[i] * w;
    }

    EmlFFT fft;
    fft.length = FEATURE_FFT_N / 2;
    fft.sin = fft_sin;
    fft.cos = fft_cos;

    EmlError err = eml_fft_fill(fft, FEATURE_FFT_N);
    if (err != EmlOk) {
        return false;
    }

    err = eml_fft_forward(fft, real, imag, FEATURE_FFT_N);
    if (err != EmlOk) {
        return false;
    }

    int spec_len = FEATURE_FFT_N / 2 + 1;
    int bins = std::min(out_len, spec_len);

    for (int k = 0; k < bins; ++k) {
        freqs[k] = static_cast<float>(k) * fs / static_cast<float>(FEATURE_FFT_N);
        mag[k] = std::sqrt(real[k] * real[k] + imag[k] * imag[k]);
    }

    return true;
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

bool mel_features_emlearn_cpp(
    const float* x,
    int n,
    float fs,
    float* mel_out,
    int n_mels
) {
    static float inout_buf[FEATURE_FFT_N];
    static float temp_buf[FEATURE_FFT_N];
    static float fft_sin[FEATURE_FFT_N / 2];
    static float fft_cos[FEATURE_FFT_N / 2];

    for (int i = 0; i < FEATURE_FFT_N; ++i) {
        inout_buf[i] = 0.0f;
        temp_buf[i] = 0.0f;
    }

    int copy_len = std::min(n, FEATURE_FFT_N);

    for (int i = 0; i < copy_len; ++i) {
        inout_buf[i] = x[i];
    }

    EmlFFT fft;
    fft.length = FEATURE_FFT_N / 2;
    fft.sin = fft_sin;
    fft.cos = fft_cos;

    EmlError err = eml_fft_fill(fft, FEATURE_FFT_N);
    if (err != EmlOk) {
        return false;
    }

    EmlAudioMel mel_params;
    mel_params.n_mels = n_mels;
    mel_params.fmin = 0.0f;
    mel_params.fmax = fs / 2.0f;
    mel_params.n_fft = FEATURE_FFT_N;
    mel_params.samplerate = static_cast<int>(fs);

    EmlVector inout;
    inout.data = inout_buf;
    inout.length = FEATURE_FFT_N;

    EmlVector temp;
    temp.data = temp_buf;
    temp.length = FEATURE_FFT_N;

    err = eml_audio_melspectrogram(mel_params, fft, inout, temp);
    if (err != EmlOk) {
        return false;
    }

    for (int i = 0; i < n_mels; ++i) {
        mel_out[i] = inout_buf[i];
    }

    return true;
}

bool mfcc_features_emlearn_cpp(
    const float* x,
    int n,
    float fs,
    float* mfcc_out,
    int n_mfcc
) {
    static float mel[N_MELS];
    static float log_mel[N_MELS];

    bool ok = mel_features_emlearn_cpp(x, n, fs, mel, N_MELS);
    if (!ok) {
        return false;
    }

    for (int i = 0; i < N_MELS; ++i) {
        log_mel[i] = std::log(mel[i] + 1e-12f);
    }

    for (int k = 0; k < n_mfcc; ++k) {
        float sum = 0.0f;

        for (int m = 0; m < N_MELS; ++m) {
            float angle = (float)M_PI * k * (m + 0.5f) / N_MELS;
            sum += log_mel[m] * std::cos(angle);
        }

        mfcc_out[k] = sum;
    }

    return true;
}

bool extract_combined_features_cpp(
    const float* x,
    int n,
    float fs,
    float* features_out
) {
    static float freqs[FEATURE_FFT_N / 2 + 1];
    static float mag[FEATURE_FFT_N / 2 + 1];
    static float mfcc[N_MFCC];
    static float mel[N_MELS];

    int spec_len = FEATURE_FFT_N / 2 + 1;

    bool spec_ok = magnitude_spectrum_fft_cpp(
        x,
        n,
        fs,
        freqs,
        mag,
        spec_len
    );

    if (!spec_ok) {
        return false;
    }

    float rms = rms_energy_cpp(x, n);
    float zcr = zero_crossing_rate_cpp(x, n);
    float centroid = spectral_centroid_cpp(freqs, mag, spec_len);
    float rolloff = spectral_rolloff_cpp(freqs, mag, spec_len, 0.85f);
    float dominant = dominant_frequency_cpp(freqs, mag, spec_len);

    float band0_500 = band_energy_cpp(freqs, mag, spec_len, 0.0f, 500.0f);
    float band500_1500 = band_energy_cpp(freqs, mag, spec_len, 500.0f, 1500.0f);
    float band1500_3000 = band_energy_cpp(freqs, mag, spec_len, 1500.0f, 3000.0f);
    float band3000_6000 = band_energy_cpp(freqs, mag, spec_len, 3000.0f, 6000.0f);

    bool mfcc_ok = mfcc_features_emlearn_cpp(x, n, fs, mfcc, N_MFCC);
    if (!mfcc_ok) {
        return false;
    }

    bool mel_ok = mel_features_emlearn_cpp(x, n, fs, mel, N_MELS);
    if (!mel_ok) {
        return false;
    }

    int idx = 0;

    features_out[idx++] = rms;
    features_out[idx++] = zcr;
    features_out[idx++] = centroid;
    features_out[idx++] = rolloff;
    features_out[idx++] = dominant;
    features_out[idx++] = band0_500;
    features_out[idx++] = band500_1500;
    features_out[idx++] = band1500_3000;
    features_out[idx++] = band3000_6000;

    for (int i = 0; i < N_MFCC; ++i) {
        features_out[idx++] = mfcc[i];
    }

    for (int i = 0; i < N_MELS; ++i) {
        features_out[idx++] = mel[i];
    }

    return true;
}