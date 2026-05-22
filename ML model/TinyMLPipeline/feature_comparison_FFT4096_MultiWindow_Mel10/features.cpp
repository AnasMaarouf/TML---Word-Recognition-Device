#include "features.h"

#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "eml_fft.h"
#include "eml_audio.h"
#include "eml_vector.h"

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

bool extract_mel_features_cpp(
    const float* x,
    int n,
    float fs,
    float* features_out
) {
    static float frame_raw[FEATURE_FFT_N];
    static float mel_sum[N_MELS];
    static float mel_frame[N_MELS];

    for (int i = 0; i < N_MELS; ++i) {
        mel_sum[i] = 0.0f;
    }

    int num_windows = 0;

    if (n <= FEATURE_FFT_N) {
        make_raw_fft_frame(x, n, 0, frame_raw);

        if (!mel_features_frame_cpp(frame_raw, fs, mel_frame, N_MELS)) {
            return false;
        }

        for (int i = 0; i < N_MELS; ++i) {
            mel_sum[i] += mel_frame[i];
        }

        num_windows = 1;
    } else {
        int last_start = n - FEATURE_FFT_N;
        int last_processed_start = -1;

        for (int start = 0; start <= last_start; start += FEATURE_FFT_HOP) {
            make_raw_fft_frame(x, n, start, frame_raw);

            if (!mel_features_frame_cpp(frame_raw, fs, mel_frame, N_MELS)) {
                return false;
            }

            for (int i = 0; i < N_MELS; ++i) {
                mel_sum[i] += mel_frame[i];
            }

            last_processed_start = start;
            num_windows++;
        }

        if (last_processed_start != last_start) {
            make_raw_fft_frame(x, n, last_start, frame_raw);

            if (!mel_features_frame_cpp(frame_raw, fs, mel_frame, N_MELS)) {
                return false;
            }

            for (int i = 0; i < N_MELS; ++i) {
                mel_sum[i] += mel_frame[i];
            }

            num_windows++;
        }
    }

    float inv = 1.0f / static_cast<float>(num_windows);

    for (int i = 0; i < N_MELS; ++i) {
        features_out[i] = mel_sum[i] * inv;
    }

    return true;
}