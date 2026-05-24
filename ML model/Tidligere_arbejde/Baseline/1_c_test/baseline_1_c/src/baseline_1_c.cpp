#include "Particle.h"
#include <stdint.h>
#include <stdio.h>

// inkluder slicet test dataset seperat, ellers kommer der overflow
#include "test_dataset_part7.h"
#include "keyword_model_simple_fft4096.h"
#include "features.h"

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

SerialLogHandler logHandler;

#define N_CLASSES 6
static constexpr int FFT_N_LOCAL = 4096;
static constexpr int SPEC_LEN = FFT_N_LOCAL / 2 + 1;

// Store arrays globalt/static i stedet for på stacken
static float freqs[SPEC_LEN];
static float mag[SPEC_LEN];

namespace {
    void run_testset() {
        const float fs = 16000.0f;
        int cm[N_CLASSES][N_CLASSES] = {0};
        int correct = 0;

        Serial.println("Starting Photon 2 offline testset evaluation...");
        Serial.println();

        for (int i = 0; i < test_num_samples; ++i) {
            const float* audio = test_audio_ptrs[i];
            const int n = test_lengths[i];
            const int true_label = test_labels[i];

            float rms = rms_energy_cpp(audio, n);
            float zcr = zero_crossing_rate_cpp(audio, n);

            magnitude_spectrum_cpp(audio, n, fs, freqs, mag, SPEC_LEN);

            float centroid = spectral_centroid_cpp(freqs, mag, SPEC_LEN);
            float rolloff = spectral_rolloff_cpp(freqs, mag, SPEC_LEN, 0.85f);
            float dominant = dominant_frequency_cpp(freqs, mag, SPEC_LEN);
            float band0_500 = band_energy_cpp(freqs, mag, SPEC_LEN, 0.0f, 500.0f);
            float band500_1500 = band_energy_cpp(freqs, mag, SPEC_LEN, 500.0f, 1500.0f);
            float band1500_3000 = band_energy_cpp(freqs, mag, SPEC_LEN, 1500.0f, 3000.0f);
            float band3000_6000 = band_energy_cpp(freqs, mag, SPEC_LEN, 3000.0f, 6000.0f);

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