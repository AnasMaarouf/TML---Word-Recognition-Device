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
        if (prev != curr) crossings++;
    }
    return static_cast<float>(crossings) / static_cast<float>(n - 1);
}

static void make_windowed_fft_frame(
    const float* x,
    int n,
    int start,
    float* frame
) {
    for (int i = 0; i < FEATURE_FFT_N; ++i) {
        frame[i] = 0.0f;
    }

    int available = n - start;
    int copy_len = std::min(available, FEATURE_FFT_N);

    for (int i = 0; i < copy_len; ++i) {
        float w = 0.5f - 0.5f * std::cos(
            2.0f * (float)M_PI * i / (FEATURE_FFT_N - 1)
        );
        frame[i] = x[start + i] * w;
    }
}

static void make_raw_fft_frame(
    const float* x,
    int n,
    int start,
    float* frame
) {
    for (int i = 0; i < FEATURE_FFT_N; ++i) {
        frame[i] = 0.0f;
    }

    int available = n - start;
    int copy_len = std::min(available, FEATURE_FFT_N);

    for (int i = 0; i < copy_len; ++i) {
        frame[i] = x[start + i];
    }
}

static bool magnitude_spectrum_frame_cpp(
    const float* frame,
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
        real[i] = frame[i];
        imag[i] = 0.0f;
    }

    EmlFFT fft;
    fft.length = FEATURE_FFT_N / 2;
    fft.sin = fft_sin;
    fft.cos = fft_cos;

    EmlError err = eml_fft_fill(fft, FEATURE_FFT_N);
    if (err != EmlOk) return false;

    err = eml_fft_forward(fft, real, imag, FEATURE_FFT_N);
    if (err != EmlOk) return false;

    int spec_len = FEATURE_FFT_N / 2 + 1;
    int bins = std::min(out_len, spec_len);

    for (int k = 0; k < bins; ++k) {
        freqs[k] = static_cast<float>(k) * fs / static_cast<float>(FEATURE_FFT_N);
        mag[k] = std::sqrt(real[k] * real[k] + imag[k] * imag[k]);
    }

    return true;
}

static void simple_features_from_spectrum(
    const float* freqs,
    const float* mag,
    int n,
    float* out
) {
    double centroid_num = 0.0;
    double centroid_den = 0.0;

    for (int i = 0; i < n; ++i) {
        centroid_num += freqs[i] * mag[i];
        centroid_den += mag[i];
    }

    out[0] = static_cast<float>(centroid_num / (centroid_den + 1e-12));

    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        total += mag[i];
    }

    double threshold = 0.85 * total;
    double cumsum = 0.0;
    out[1] = freqs[n - 1];

    for (int i = 0; i < n; ++i) {
        cumsum += mag[i];
        if (cumsum >= threshold) {
            out[1] = freqs[i];
            break;
        }
    }

    int dom_idx = 1;
    for (int i = 2; i < n; ++i) {
        if (mag[i] > mag[dom_idx]) {
            dom_idx = i;
        }
    }

    out[2] = freqs[dom_idx];

    double band0_500 = 0.0;
    double band500_1500 = 0.0;
    double band1500_3000 = 0.0;
    double band3000_6000 = 0.0;

    for (int i = 0; i < n; ++i) {
        float f = freqs[i];
        float p = mag[i] * mag[i];

        if (f >= 0.0f && f < 500.0f) band0_500 += p;
        else if (f >= 500.0f && f < 1500.0f) band500_1500 += p;
        else if (f >= 1500.0f && f < 3000.0f) band1500_3000 += p;
        else if (f >= 3000.0f && f < 6000.0f) band3000_6000 += p;
    }

    out[3] = static_cast<float>(band0_500);
    out[4] = static_cast<float>(band500_1500);
    out[5] = static_cast<float>(band1500_3000);
    out[6] = static_cast<float>(band3000_6000);
}

static bool mel_features_frame_cpp(
    const float* raw_frame,
    float fs,
    float* mel_out,
    int n_mels
) {
    static float inout_buf[FEATURE_FFT_N];
    static float temp_buf[FEATURE_FFT_N];
    static float fft_sin[FEATURE_FFT_N / 2];
    static float fft_cos[FEATURE_FFT_N / 2];

    for (int i = 0; i < FEATURE_FFT_N; ++i) {
        inout_buf[i] = raw_frame[i];
        temp_buf[i] = 0.0f;
    }

    EmlFFT fft;
    fft.length = FEATURE_FFT_N / 2;
    fft.sin = fft_sin;
    fft.cos = fft_cos;

    EmlError err = eml_fft_fill(fft, FEATURE_FFT_N);
    if (err != EmlOk) return false;

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
    if (err != EmlOk) return false;

    for (int i = 0; i < n_mels; ++i) {
        mel_out[i] = inout_buf[i];
    }

    return true;
}

static bool process_one_window(
    const float* x,
    int n,
    int start,
    float fs,
    float* simple_sum,
    float* mel_sum
) {
    static float frame_windowed[FEATURE_FFT_N];
    static float frame_raw[FEATURE_FFT_N];

    static float freqs[FEATURE_FFT_N / 2 + 1];
    static float mag[FEATURE_FFT_N / 2 + 1];

    static float simple_frame[7];
    static float mel_frame[N_MELS];

    make_windowed_fft_frame(x, n, start, frame_windowed);
    make_raw_fft_frame(x, n, start, frame_raw);

    if (!magnitude_spectrum_frame_cpp(
        frame_windowed,
        fs,
        freqs,
        mag,
        FEATURE_FFT_N / 2 + 1
    )) {
        return false;
    }

    simple_features_from_spectrum(
        freqs,
        mag,
        FEATURE_FFT_N / 2 + 1,
        simple_frame
    );

    if (!mel_features_frame_cpp(
        frame_raw,
        fs,
        mel_frame,
        N_MELS
    )) {
        return false;
    }

    for (int i = 0; i < 7; ++i) {
        simple_sum[i] += simple_frame[i];
    }

    for (int i = 0; i < N_MELS; ++i) {
        mel_sum[i] += mel_frame[i];
    }

    return true;
}

bool extract_simple_mel_features_cpp(
    const float* x,
    int n,
    float fs,
    float* features_out
) {
    static float simple_sum[7];
    static float mel_sum[N_MELS];

    for (int i = 0; i < 7; ++i) simple_sum[i] = 0.0f;
    for (int i = 0; i < N_MELS; ++i) mel_sum[i] = 0.0f;

    int num_windows = 0;

    if (n <= FEATURE_FFT_N) {
        if (!process_one_window(x, n, 0, fs, simple_sum, mel_sum)) {
            return false;
        }
        num_windows = 1;
    } else {
        int last_start = n - FEATURE_FFT_N;
        int last_processed_start = -1;

        for (int start = 0; start <= last_start; start += FEATURE_FFT_HOP) {
            if (!process_one_window(x, n, start, fs, simple_sum, mel_sum)) {
                return false;
            }

            last_processed_start = start;
            num_windows++;
        }

        if (last_processed_start != last_start) {
            if (!process_one_window(x, n, last_start, fs, simple_sum, mel_sum)) {
                return false;
            }

            num_windows++;
        }
    }

    float inv = 1.0f / static_cast<float>(num_windows);

    int idx = 0;

    features_out[idx++] = rms_energy_cpp(x, n);
    features_out[idx++] = zero_crossing_rate_cpp(x, n);

    for (int i = 0; i < 7; ++i) {
        features_out[idx++] = simple_sum[i] * inv;
    }

    for (int i = 0; i < N_MELS; ++i) {
        features_out[idx++] = mel_sum[i] * inv;
    }

    return true;
}