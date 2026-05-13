/* 

NEDENSTÅENDE ER FORNUFTIG NOK TIL FORSKEL MELLEM LAKS OG UNKNOWN

#include "Particle.h"
#include "Microphone_PDM.h"

#include "keyword_model_rf_mel10_fft4096.h"
#include "features.h"

#include <cmath>

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

// =========================
// CONFIG
// =========================
static constexpr float FS = 16000.0f;
static constexpr int N_CLASSES = 6;

static constexpr int CLIP_SAMPLES = 16000;
static constexpr int PRE_SAMPLES  = 4800; // 0.3s

static constexpr unsigned long DETECTION_COOLDOWN_MS = 500;

static constexpr float REARM_THRESHOLD = 0.06f;
static constexpr unsigned long REARM_SILENCE_MS = 500;

static constexpr int RING_BUFFER_SIZE = PRE_SAMPLES;

static constexpr float THRESH_MULT = 8.0f;
static constexpr float MIN_START_THRESHOLD = 0.07f;
static constexpr float NOISE_FLOOR_ALPHA = 0.02f;
static constexpr int START_CONSECUTIVE_CHUNKS = 2;
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
static uint16_t ringWriteIndex = 0;
static bool ringFilled = false;

static int16_t clipBuffer[CLIP_SAMPLES];
static uint16_t clipIndex = 0;

static float features[MEL_FEATURES_N];

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
static uint8_t aboveThresholdCount = 0;

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
        uint16_t idx = ringWriteIndex;

        for (uint16_t i = 0; i < PRE_SAMPLES; ++i) {
            clipBuffer[i] = ringBuffer[idx];

            idx++;
            if (idx >= RING_BUFFER_SIZE) {
                idx = 0;
            }
        }
    } else {
        uint16_t valid = ringWriteIndex;
        uint16_t pad = PRE_SAMPLES - valid;

        for (uint16_t i = 0; i < pad; ++i) {
            clipBuffer[i] = 0;
        }

        for (uint16_t i = 0; i < valid; ++i) {
            clipBuffer[pad + i] = ringBuffer[i];
        }
    }

    clipIndex = PRE_SAMPLES;
}

int classify_fixed_clip(const int16_t* audio, int n) {
    int clip_len = (n < CLIP_SAMPLES) ? n : CLIP_SAMPLES;

    bool ok = extract_mel_features_i16_cpp(
        audio,
        clip_len,
        FS,
        features
    );

    if (!ok) {
        Serial.println("Feature extraction failed");
        return -1;
    }

    float proba[N_CLASSES];

    keyword_model_predict_proba(
        features,
        MEL_FEATURES_N,
        proba,
        N_CLASSES
    );

    int pred = 0;
    float best = proba[0];

    for (int i = 1; i < N_CLASSES; ++i) {
        if (proba[i] > best) {
            best = proba[i];
            pred = i;
        }
    }

    // Reject hvis modellen ikke er sikker nok
    if (best < 0.65f) {
        pred = 5; // unknown
    }

    if (pred >= 0 && pred < N_CLASSES) {
        Serial.printf(
        "CLIP DETECTED -> pred=%d label=%s conf=%.3f mel1=%.6f mel2=%.6f mel3=%.6f\r\n",
        pred,
        CLASS_NAMES[pred],
        best,
        features[0],
        features[1],
        features[2]
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

    static float noiseFloor = 0.02f;

    if (captureState == STATE_IDLE) {
        // Opdater kun noise floor når vi IKKE er over trigger
        float adaptiveThreshold = noiseFloor * THRESH_MULT;

        if (adaptiveThreshold < MIN_START_THRESHOLD) {
            adaptiveThreshold = MIN_START_THRESHOLD;
        }

        if (chunkRms < adaptiveThreshold) {
            noiseFloor =
                (1.0f - NOISE_FLOOR_ALPHA) * noiseFloor +
                NOISE_FLOOR_ALPHA * chunkRms;

            aboveThresholdCount = 0;
        } else if ((now - lastDetectionMs) > DETECTION_COOLDOWN_MS) {
            aboveThresholdCount++;

            if (aboveThresholdCount >= START_CONSECUTIVE_CHUNKS) {
                Serial.printf(
                    "TRIGGER rms=%.4f noise=%.4f thr=%.4f count=%d\r\n",
                    chunkRms,
                    noiseFloor,
                    adaptiveThreshold,
                    aboveThresholdCount
                );

                copy_preroll_from_ring_to_clip();
                captureState = STATE_CAPTURING;
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

    Serial.println("Starting live KWS: RF Mel10 multi-window FFT4096");

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

#include "keyword_model_rf_mel10_fft4096.h"
#include "features.h"

#include <cmath>

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

// =========================
// CONFIG
// =========================
static constexpr float FS = 16000.0f;
static constexpr int N_CLASSES = 6;

static constexpr int CLIP_SAMPLES = 16000;
static constexpr int PRE_SAMPLES  = 3200; // 0.2s

static constexpr unsigned long DETECTION_COOLDOWN_MS = 500;

static constexpr float REARM_THRESHOLD = 0.06f;
static constexpr unsigned long REARM_SILENCE_MS = 500;

static constexpr int RING_BUFFER_SIZE = PRE_SAMPLES;

static constexpr float THRESH_MULT = 6.0f;
static constexpr float MIN_START_THRESHOLD = 0.04f;
static constexpr float NOISE_FLOOR_ALPHA = 0.02f;
static constexpr int START_CONSECUTIVE_CHUNKS = 3;
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
static uint16_t ringWriteIndex = 0;
static bool ringFilled = false;

static int16_t clipBuffer[CLIP_SAMPLES];
static uint16_t clipIndex = 0;

static float features[MEL_FEATURES_N];

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
static uint8_t aboveThresholdCount = 0;

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
        uint16_t idx = ringWriteIndex;

        for (uint16_t i = 0; i < PRE_SAMPLES; ++i) {
            clipBuffer[i] = ringBuffer[idx];

            idx++;
            if (idx >= RING_BUFFER_SIZE) {
                idx = 0;
            }
        }
    } else {
        uint16_t valid = ringWriteIndex;
        uint16_t pad = PRE_SAMPLES - valid;

        for (uint16_t i = 0; i < pad; ++i) {
            clipBuffer[i] = 0;
        }

        for (uint16_t i = 0; i < valid; ++i) {
            clipBuffer[pad + i] = ringBuffer[i];
        }
    }

    clipIndex = PRE_SAMPLES;
}

int classify_fixed_clip(const int16_t* audio, int n) {
    int clip_len = (n < CLIP_SAMPLES) ? n : CLIP_SAMPLES;

    bool ok = extract_mel_features_i16_cpp(
        audio,
        clip_len,
        FS,
        features
    );

    if (!ok) {
        Serial.println("Feature extraction failed");
        return -1;
    }

    float proba[N_CLASSES];

    keyword_model_predict_proba(
        features,
        MEL_FEATURES_N,
        proba,
        N_CLASSES
    );

    int pred = 0;
    float best = proba[0];

    for (int i = 1; i < N_CLASSES; ++i) {
        if (proba[i] > best) {
            best = proba[i];
            pred = i;
        }
    }

    if (best < 0.45f) {
        pred = 5;
    }

    Serial.printf(
        "CLIP DETECTED -> pred=%d label=%s conf=%.3f\r\n",
        pred,
        CLASS_NAMES[pred],
        best
    );

    Serial.printf(
        "PROBA -> får=%.3f ged=%.3f hest=%.3f laks=%.3f ulv=%.3f unknown=%.3f\r\n",
        proba[0],
        proba[1],
        proba[2],
        proba[3],
        proba[4],
        proba[5]
    );

    Serial.printf(
        "MELS -> %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\r\n",
        features[0],
        features[1],
        features[2],
        features[3],
        features[4],
        features[5],
        features[6],
        features[7],
        features[8],
        features[9]
    );

    return pred;
}

void process_audio_chunk(const int16_t* src, size_t numSamples) {
    const unsigned long now = millis();
    const float chunkRms = rms_int16(src, numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        push_ring_sample(src[i]);
    }

    static float noiseFloor = 0.02f;

    if (captureState == STATE_IDLE) {
        // Opdater kun noise floor når vi IKKE er over trigger
        float adaptiveThreshold = noiseFloor * THRESH_MULT;

        if (adaptiveThreshold < MIN_START_THRESHOLD) {
            adaptiveThreshold = MIN_START_THRESHOLD;
        }

        if (chunkRms < adaptiveThreshold) {
            noiseFloor =
                (1.0f - NOISE_FLOOR_ALPHA) * noiseFloor +
                NOISE_FLOOR_ALPHA * chunkRms;

            aboveThresholdCount = 0;
        } else if ((now - lastDetectionMs) > DETECTION_COOLDOWN_MS) {
            aboveThresholdCount++;

            if (aboveThresholdCount >= START_CONSECUTIVE_CHUNKS) {
                Serial.printf(
                    "TRIGGER rms=%.4f noise=%.4f thr=%.4f count=%d\r\n",
                    chunkRms,
                    noiseFloor,
                    adaptiveThreshold,
                    aboveThresholdCount
                );

                copy_preroll_from_ring_to_clip();
                captureState = STATE_CAPTURING;
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

    Serial.println("Starting live KWS: RF Mel10 multi-window FFT4096");

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