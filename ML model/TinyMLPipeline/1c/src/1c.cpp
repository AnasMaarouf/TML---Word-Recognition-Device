#include "Particle.h"
#include <stdint.h>
#include <stdio.h>

#include "test_dataset_part11.h"
#include "keyword_model_rf_mel10_fft4096.h"
#include "features.h"

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

SerialLogHandler logHandler;

#define N_CLASSES 6
#define N_FEATURES MEL_FEATURES_N

namespace {
    void run_testset() {
        const float fs = 16000.0f;

        int cm[N_CLASSES][N_CLASSES] = {0};
        int correct = 0;
        int valid_samples = 0;
        int failed_features = 0;

        Serial.println("Starting Photon 2 offline testset evaluation: RF + Mel10");
        Serial.println();

        for (int i = 0; i < test_num_samples; ++i) {
            const int16_t* audio = test_audio_ptrs[i];
            const int n = test_lengths[i];
            const int true_label = test_labels[i];

            float features[N_FEATURES];

            bool ok = extract_mel_features_i16_cpp(
                audio,
                n,
                fs,
                features
            );

            if (!ok) {
                Serial.printf("sample %d: feature extraction failed\r\n", i);
                failed_features++;
                delay(1);
                continue;
            }

            int pred = keyword_model_predict(features, N_FEATURES);

            if (pred >= 0 && pred < N_CLASSES) {
                cm[true_label][pred]++;
            }

            if (pred == true_label) {
                correct++;
            }

            valid_samples++;

            Serial.printf(
                "sample %d: true=%d pred=%d\r\n",
                i,
                true_label,
                pred
            );

            delay(1);
        }

        Serial.println();

        if (valid_samples > 0) {
            Serial.printf(
                "Accuracy: %.4f (%d/%d)\r\n",
                (double)correct / (double)valid_samples,
                correct,
                valid_samples
            );
        } else {
            Serial.println("Accuracy: N/A, no valid samples");
        }

        Serial.printf("Feature extraction failures: %d\r\n", failed_features);

        Serial.println();
        Serial.println("Confusion matrix:");
        for (int r = 0; r < N_CLASSES; ++r) {
            for (int c = 0; c < N_CLASSES; ++c) {
                Serial.printf("%4d ", cm[r][c]);
            }
            Serial.println();
        }

        Serial.println();
        Serial.println("Done.");
    }
}

void setup() {
    Serial.begin(115200);
    waitFor(Serial.isConnected, 10000);
    delay(2000);

    run_testset();
}

void loop() {
}