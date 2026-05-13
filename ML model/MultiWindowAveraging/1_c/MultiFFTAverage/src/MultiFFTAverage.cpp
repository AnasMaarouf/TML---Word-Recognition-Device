#include "Particle.h"
#include <stdint.h>
#include <stdio.h>

#include "test_dataset_part0.h"
#include "keyword_model_rf_simple_mel_multiwindow_fft4096.h"
#include "features.h"

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

SerialLogHandler logHandler;

#define N_CLASSES 6
#define N_FEATURES SIMPLE_MEL_FEATURES_N
#define MAX_AUDIO_SAMPLES 16000

static float features[N_FEATURES];
static float audio_float[MAX_AUDIO_SAMPLES];

namespace {
    void run_testset() {
        const float fs = 16000.0f;

        int cm[N_CLASSES][N_CLASSES] = {0};
        int correct = 0;

        Serial.println("Starting Photon 2 offline int16 testset evaluation...");
        Serial.println("Model: RF simple_mel multi-window FFT4096");
        Serial.println();

        for (int i = 0; i < test_num_samples; ++i) {
            const int16_t* audio_i16 = test_audio_ptrs[i];
            const int n = test_lengths[i];
            const int true_label = test_labels[i];

            if (n > MAX_AUDIO_SAMPLES) {
                Serial.printf("sample %d: too long n=%d\r\n", i, n);
                continue;
            }

            for (int j = 0; j < n; ++j) {
                audio_float[j] = (float)audio_i16[j] / 32767.0f;
            }

            bool ok = extract_simple_mel_features_cpp(
                audio_float,
                n,
                fs,
                features
            );

            if (!ok) {
                Serial.printf("sample %d: feature extraction failed\r\n", i);
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

            Serial.printf("sample %d: true=%d pred=%d\r\n", i, true_label, pred);
            delay(1);
        }

        Serial.println();
        Serial.printf("Accuracy: %.4f (%d/%ld)\r\n",
            (double)correct / (double)test_num_samples,
            correct,
            (long)test_num_samples);

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