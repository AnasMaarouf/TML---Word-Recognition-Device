#include <stdio.h>
#include <stdint.h>

#include "test_dataset.h"
#include "keyword_model_simple_fft4096.h"
#include "features.h"

#define N_CLASSES 6

int main(void)
{
    const float fs = 16000.0f;
    int cm[N_CLASSES][N_CLASSES] = {0};
    int correct = 0;

    for (int i = 0; i < test_num_samples; ++i) {
        const float* audio = test_audio_ptrs[i];
        const int n = test_lengths[i];
        const int true_label = test_labels[i];

        static constexpr int FFT_N = 4096;
        const int spec_len = FFT_N / 2 + 1;
        float freqs[spec_len];
        float mag[spec_len];

        float rms = rms_energy_cpp(audio, n);
        float zcr = zero_crossing_rate_cpp(audio, n);

        magnitude_spectrum_cpp(audio, n, fs, freqs, mag, spec_len);

        float centroid = spectral_centroid_cpp(freqs, mag, spec_len);
        float rolloff = spectral_rolloff_cpp(freqs, mag, spec_len, 0.85f);
        float dominant = dominant_frequency_cpp(freqs, mag, spec_len);
        float band0_500 = band_energy_cpp(freqs, mag, spec_len, 0.0f, 500.0f);
        float band500_1500 = band_energy_cpp(freqs, mag, spec_len, 500.0f, 1500.0f);
        float band1500_3000 = band_energy_cpp(freqs, mag, spec_len, 1500.0f, 3000.0f);
        float band3000_6000 = band_energy_cpp(freqs, mag, spec_len, 3000.0f, 6000.0f);

        float features[9] = {
            rms,
            zcr,
            centroid,
            rolloff,
            dominant,
            band0_500,
            band500_1500,
            band1500_3000,
            band3000_6000
        };

        int pred = keyword_model_predict(features, 9);

        if (pred >= 0 && pred < N_CLASSES) {
            cm[true_label][pred]++;
        }

        if (pred == true_label) {
            correct++;
        }

        printf("sample %d: true=%d pred=%d\n", i, true_label, pred);
    }

    printf("\nAccuracy: %.4f (%d/%d)\n",
           (double)correct / (double)test_num_samples,
           correct,
           test_num_samples);

    printf("\nConfusion matrix:\n");
    for (int r = 0; r < N_CLASSES; ++r) {
        for (int c = 0; c < N_CLASSES; ++c) {
            printf("%4d ", cm[r][c]);
        }
        printf("\n");
    }

    return 0;
}