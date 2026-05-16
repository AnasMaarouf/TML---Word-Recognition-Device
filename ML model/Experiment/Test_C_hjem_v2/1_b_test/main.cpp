#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <vector>

#include "test_dataset.h"
#include "features.h"

#include "keyword_model_knn_combined_fft4096.h"
#include "keyword_scaler_knn_combined_fft4096.h"

#define N_CLASSES 6
#define N_FEATURES 42

#define SINGLE_SAMPLE_INDEX 0

int main(void)
{
    const float fs = 16000.0f;

    int cm[N_CLASSES][N_CLASSES] = {0};
    int correct = 0;

    std::vector<EmlNeighborsDistanceItem> distances(keyword_model.n_items);

    for (int i = 0; i < test_num_samples; ++i) {
        const float* audio = test_audio_ptrs[i];
        const int n = test_lengths[i];
        const int true_label = test_labels[i];

        float features[N_FEATURES];

        bool ok = extract_combined_features_cpp(
            audio,
            n,
            fs,
            features
        );

        if (!ok) {
            printf("sample %d: feature extraction failed\n", i);
            continue;
        }

        keyword_scale_features(features);

        int16_t features_i16[N_FEATURES];
        keyword_quantize_features(features, features_i16);

        int16_t pred_i16 = -1;

        EmlError err = eml_neighbors_predict(
            &keyword_model,
            features_i16,
            N_FEATURES,
            distances.data(),
            (int)distances.size(),
            &pred_i16
        );

        if (err != EmlOk) {
            printf("sample %d: prediction failed, err=%d\n", i, err);
            continue;
        }

        int pred = (int)pred_i16;

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