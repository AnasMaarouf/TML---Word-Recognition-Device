#include <stdio.h>
#include <stdint.h>

#include "test_dataset.h"
#include "keyword_model_rf_mel10_fft4096.h"
#include "features.h"

#define N_CLASSES 6
#define N_FEATURES MEL_FEATURES_N

int main(void)
{
    const float fs = 16000.0f;

    int cm[N_CLASSES][N_CLASSES] = {0};
    int correct = 0;

    for (int i = 0; i < test_num_samples; ++i) {
        const float* audio = test_audio_ptrs[i];
        const int n = test_lengths[i];
        const int true_label = test_labels[i];

        float features[N_FEATURES];

        bool ok = extract_mel_features_cpp(
            audio,
            n,
            fs,
            features
        );

        if (!ok) {
            printf("sample %d: feature extraction failed\n", i);
            continue;
        }

        int pred = keyword_model_predict(features, N_FEATURES);

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