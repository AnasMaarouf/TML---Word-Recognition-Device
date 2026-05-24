/*
#include "Particle.h"
#include "Microphone_PDM.h"

#include "keyword_model_simple_fft4096.h"
#include "features.h"

#include <cmath>

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

// =========================
// CONFIG
// =========================
static constexpr float FS = 16000.0f;
static constexpr int N_CLASSES = 6;

static constexpr int FFT_N_LOCAL = 4096;
static constexpr int SPEC_LEN = FFT_N_LOCAL / 2 + 1;

// Mimic Python slicing
static constexpr int CLIP_SAMPLES = 16000;   // 1.00 s
static constexpr int PRE_SAMPLES  = 3200;    // 0.20 s

// Trigger / cooldown / rearm
static constexpr float START_THRESHOLD = 0.10f;
static constexpr unsigned long DETECTION_COOLDOWN_MS = 700;

static constexpr float REARM_THRESHOLD = 0.05f;
static constexpr int START_CONSECUTIVE_CHUNKS = 4;
static constexpr int QUIET_CONSECUTIVE_CHUNKS = 8;

// Ring buffer til preroll
static constexpr int RING_BUFFER_SIZE = PRE_SAMPLES;

// =========================
// LABELS
// =========================
static const char* CLASS_NAMES[N_CLASSES] = {
    "får", "ged", "hest", "laks", "ulv", "unknown"
};

// =========================
// BUFFERS
// =========================
static int16_t ringBuffer[RING_BUFFER_SIZE];
static int ringWriteIndex = 0;
static bool ringFilled = false;

static int16_t clipBuffer[CLIP_SAMPLES];
static int clipIndex = 0;

static float fftInput[FFT_N_LOCAL];
static float freqs[SPEC_LEN];
static float mag[SPEC_LEN];

// =========================
// STATE / COUNTERS
// =========================
enum CaptureState {
    STATE_IDLE = 0,
    STATE_CAPTURING,
    STATE_WAIT_FOR_SILENCE
};

static CaptureState captureState = STATE_IDLE;

static unsigned long lastDetectionMs = 0;
static int aboveThresholdCount = 0;
static int quietCount = 0;

// =========================
// HELPERS
// =========================
float rms_int16(const int16_t* x, size_t n) {
    if (n == 0) return 0.0f;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float v = static_cast<float>(x[i]) / 32768.0f;
        sum += v * v;
    }
    return std::sqrt(sum / n + 1e-12);
}

void push_ring_sample(int16_t s) {
    ringBuffer[ringWriteIndex] = s;
    ringWriteIndex++;

    if (ringWriteIndex >= RING_BUFFER_SIZE) {
        ringWriteIndex = 0;
        ringFilled = true;
    }
}

void copy_preroll_from_ring_to_clip() {
    if (ringFilled) {
        int idx = ringWriteIndex;
        for (int i = 0; i < PRE_SAMPLES; ++i) {
            clipBuffer[i] = ringBuffer[idx];
            idx++;
            if (idx >= RING_BUFFER_SIZE) {
                idx = 0;
            }
        }
    } else {
        int valid = ringWriteIndex;
        int pad = PRE_SAMPLES - valid;

        for (int i = 0; i < pad; ++i) {
            clipBuffer[i] = 0;
        }
        for (int i = 0; i < valid; ++i) {
            clipBuffer[pad + i] = ringBuffer[i];
        }
    }

    clipIndex = PRE_SAMPLES;
}

int classify_fixed_clip(const int16_t* audio, int n) {
    int clip_len = (n < CLIP_SAMPLES) ? n : CLIP_SAMPLES;

    float max_abs = 1.0f;
    for (int i = 0; i < clip_len; ++i) {
        float a = fabsf((float)audio[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    const float norm_scale = 1.0f / max_abs;

    double sumsq = 0.0;
    for (int i = 0; i < clip_len; ++i) {
        float v = ((float)audio[i]) * norm_scale;
        sumsq += v * v;
    }
    float rms = sqrtf((float)(sumsq / CLIP_SAMPLES) + 1e-12f);

    int crossings = 0;
    for (int i = 1; i < clip_len; ++i) {
        bool prev = audio[i - 1] < 0;
        bool curr = audio[i] < 0;
        if (prev != curr) {
            crossings++;
        }
    }
    float zcr = (clip_len > 1)
        ? ((float)crossings / (float)(CLIP_SAMPLES - 1))
        : 0.0f;

    for (int i = 0; i < FFT_N_LOCAL; ++i) {
        fftInput[i] = 0.0f;
    }

    int fft_copy_len = (clip_len < FFT_N_LOCAL) ? clip_len : FFT_N_LOCAL;
    for (int i = 0; i < fft_copy_len; ++i) {
        fftInput[i] = ((float)audio[i]) * norm_scale;
    }

    magnitude_spectrum_cpp(fftInput, FFT_N_LOCAL, FS, freqs, mag, SPEC_LEN);

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
        Serial.printf(
            "CLIP DETECTED -> pred=%d label=%s rms=%.4f zcr=%.4f centroid=%.1f rolloff=%.1f dom=%.1f\r\n",
            pred,
            CLASS_NAMES[pred],
            rms,
            zcr,
            centroid,
            rolloff,
            dominant
        );
    } else {
        Serial.printf("CLIP DETECTED -> pred=%d\r\n", pred);
    }

    return pred;
}

void process_audio_chunk(const int16_t* src, size_t numSamples) {
    const unsigned long now = millis();
    const float chunkRms = rms_int16(src, numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        push_ring_sample(src[i]);
    }

    if (captureState == STATE_IDLE) {
        if ((now - lastDetectionMs) > DETECTION_COOLDOWN_MS) {
            if (chunkRms >= START_THRESHOLD) {
                aboveThresholdCount++;

                if (aboveThresholdCount >= START_CONSECUTIVE_CHUNKS) {
                    Serial.printf("TRIGGER rms=%.4f count=%d\r\n", chunkRms, aboveThresholdCount);
                    copy_preroll_from_ring_to_clip();
                    captureState = STATE_CAPTURING;
                    aboveThresholdCount = 0;
                    quietCount = 0;
                }
            } else {
                aboveThresholdCount = 0;
            }
        }
        return;
    }

    if (captureState == STATE_CAPTURING) {
        for (size_t i = 0; i < numSamples; ++i) {
            if (clipIndex < CLIP_SAMPLES) {
                clipBuffer[clipIndex++] = src[i];
            }
        }

        if (clipIndex >= CLIP_SAMPLES) {
            Serial.printf("CAPTURE DONE len=%d\r\n", clipIndex);
            classify_fixed_clip(clipBuffer, clipIndex);

            clipIndex = 0;
            lastDetectionMs = now;
            aboveThresholdCount = 0;
            quietCount = 0;
            captureState = STATE_WAIT_FOR_SILENCE;
        }
        return;
    }

    if (captureState == STATE_WAIT_FOR_SILENCE) {
        if (chunkRms < REARM_THRESHOLD) {
            quietCount++;

            if (quietCount >= QUIET_CONSECUTIVE_CHUNKS) {
                Serial.printf("REARMED rms=%.4f quietCount=%d\r\n", chunkRms, quietCount);
                aboveThresholdCount = 0;
                quietCount = 0;
                captureState = STATE_IDLE;
            }
        } else {
            quietCount = 0;
        }
        return;
    }
}

void setup() {
    Serial.begin(115200);
    waitFor(Serial.isConnected, 10000);
    delay(2000);

    Serial.println("Starting live KWS with trigger + 0.2s preroll + 1.0s fixed capture...");

    int err = Microphone_PDM::instance()
        .withOutputSize(Microphone_PDM::OutputSize::SIGNED_16)
        .withRange(Microphone_PDM::Range::RANGE_2048)
        .withSampleRate(16000)
        .init();

    if (err) {
        Serial.printf("PDM decoder init err=%d\r\n", err);
        return;
    }

    err = Microphone_PDM::instance().start();
    if (err) {
        Serial.printf("PDM decoder start err=%d\r\n", err);
        return;
    }
}

void loop() {
    Microphone_PDM::instance().noCopySamples([](void* pSamples, size_t numSamples) {
        const int16_t* s = static_cast<const int16_t*>(pSamples);
        process_audio_chunk(s, numSamples);
    });

    delay(1);
}
*/

#include "Particle.h"
#include "Microphone_PDM.h"

#include "keyword_model_rf_simple_mel_multiwindow_fft4096.h"
#include "features.h"

#include <cmath>

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

// =========================
// CONFIG
// =========================
static constexpr float FS = 16000.0f;
static constexpr int N_CLASSES = 6;

static constexpr int CLIP_SAMPLES = 16000;   // 1.0 s
static constexpr int PRE_SAMPLES  = 3200;    // 0.2 s

static constexpr float START_THRESHOLD = 0.05f;
static constexpr unsigned long DETECTION_COOLDOWN_MS = 500;

static constexpr float REARM_THRESHOLD = 0.03f;
static constexpr unsigned long REARM_SILENCE_MS = 150;

static constexpr int RING_BUFFER_SIZE = PRE_SAMPLES;

// =========================
// LABELS
// =========================
static const char* CLASS_NAMES[N_CLASSES] = {
    "får", "ged", "hest", "laks", "ulv", "unknown"
};

// =========================
// BUFFERS
// =========================
static int16_t ringBuffer[RING_BUFFER_SIZE];
static int ringWriteIndex = 0;
static bool ringFilled = false;

static int16_t clipBuffer[CLIP_SAMPLES];
static int clipIndex = 0;

static float audioFloat[CLIP_SAMPLES];
static float features[SIMPLE_MEL_FEATURES_N];

// =========================
// STATE
// =========================
enum CaptureState {
    STATE_IDLE = 0,
    STATE_CAPTURING,
    STATE_WAIT_FOR_SILENCE
};

static CaptureState captureState = STATE_IDLE;

static unsigned long lastDetectionMs = 0;
static unsigned long lastBelowRearmMs = 0;

// =========================
// HELPERS
// =========================
float rms_int16(const int16_t* x, size_t n) {
    if (n == 0) return 0.0f;

    double sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float v = static_cast<float>(x[i]) / 32768.0f;
        sum += v * v;
    }

    return std::sqrt(sum / n + 1e-12);
}

void push_ring_sample(int16_t s) {
    ringBuffer[ringWriteIndex] = s;
    ringWriteIndex++;

    if (ringWriteIndex >= RING_BUFFER_SIZE) {
        ringWriteIndex = 0;
        ringFilled = true;
    }
}

void copy_preroll_from_ring_to_clip() {
    if (ringFilled) {
        int idx = ringWriteIndex;

        for (int i = 0; i < PRE_SAMPLES; ++i) {
            clipBuffer[i] = ringBuffer[idx];

            idx++;
            if (idx >= RING_BUFFER_SIZE) {
                idx = 0;
            }
        }
    } else {
        int valid = ringWriteIndex;
        int pad = PRE_SAMPLES - valid;

        for (int i = 0; i < pad; ++i) {
            clipBuffer[i] = 0;
        }

        for (int i = 0; i < valid; ++i) {
            clipBuffer[pad + i] = ringBuffer[i];
        }
    }

    clipIndex = PRE_SAMPLES;
}

int classify_fixed_clip(const int16_t* audio, int n) {
    int clip_len = (n < CLIP_SAMPLES) ? n : CLIP_SAMPLES;

    float max_abs = 1.0f;

    for (int i = 0; i < clip_len; ++i) {
        float a = fabsf((float)audio[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    for (int i = 0; i < clip_len; ++i) {
        audioFloat[i] = ((float)audio[i]) / max_abs;
    }

    bool ok = extract_simple_mel_features_cpp(
        audioFloat,
        clip_len,
        FS,
        features
    );

    if (!ok) {
        Serial.println("Feature extraction failed");
        return -1;
    }

    int pred = keyword_model_predict(features, SIMPLE_MEL_FEATURES_N);

    if (pred >= 0 && pred < N_CLASSES) {
        Serial.printf(
            "CLIP DETECTED -> pred=%d label=%s rms=%.4f zcr=%.4f centroid=%.1f rolloff=%.1f dom=%.1f\r\n",
            pred,
            CLASS_NAMES[pred],
            features[0],
            features[1],
            features[2],
            features[3],
            features[4]
        );
    } else {
        Serial.printf("CLIP DETECTED -> pred=%d\r\n", pred);
    }

    return pred;
}

void process_audio_chunk(const int16_t* src, size_t numSamples) {
    const unsigned long now = millis();
    const float chunkRms = rms_int16(src, numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        push_ring_sample(src[i]);
    }

    if (captureState == STATE_IDLE) {
        if (chunkRms >= START_THRESHOLD &&
            (now - lastDetectionMs) > DETECTION_COOLDOWN_MS) {

            Serial.printf("TRIGGER rms=%.4f\r\n", chunkRms);

            copy_preroll_from_ring_to_clip();
            captureState = STATE_CAPTURING;
        }

        return;
    }

    if (captureState == STATE_CAPTURING) {
        for (size_t i = 0; i < numSamples; ++i) {
            if (clipIndex < CLIP_SAMPLES) {
                clipBuffer[clipIndex++] = src[i];
            }
        }

        if (clipIndex >= CLIP_SAMPLES) {
            Serial.printf("CAPTURE DONE len=%d\r\n", clipIndex);

            classify_fixed_clip(clipBuffer, clipIndex);

            clipIndex = 0;
            lastDetectionMs = now;
            lastBelowRearmMs = now;
            captureState = STATE_WAIT_FOR_SILENCE;
        }

        return;
    }

    if (captureState == STATE_WAIT_FOR_SILENCE) {
        if (chunkRms < REARM_THRESHOLD) {
            if ((now - lastBelowRearmMs) >= REARM_SILENCE_MS) {
                Serial.println("REARMED");
                captureState = STATE_IDLE;
            }
        } else {
            lastBelowRearmMs = now;
        }

        return;
    }
}

void setup() {
    Serial.begin(115200);
    waitFor(Serial.isConnected, 10000);
    delay(2000);

    Serial.println("Starting live KWS: RF simple_mel multi-window FFT4096");

    int err = Microphone_PDM::instance()
        .withOutputSize(Microphone_PDM::OutputSize::SIGNED_16)
        .withRange(Microphone_PDM::Range::RANGE_2048)
        .withSampleRate(16000)
        .init();

    if (err) {
        Serial.printf("PDM decoder init err=%d\r\n", err);
        return;
    }

    err = Microphone_PDM::instance().start();

    if (err) {
        Serial.printf("PDM decoder start err=%d\r\n", err);
        return;
    }
}

void loop() {
    Microphone_PDM::instance().noCopySamples([](void* pSamples, size_t numSamples) {
        const int16_t* s = static_cast<const int16_t*>(pSamples);
        process_audio_chunk(s, numSamples);
    });

    delay(1);
}