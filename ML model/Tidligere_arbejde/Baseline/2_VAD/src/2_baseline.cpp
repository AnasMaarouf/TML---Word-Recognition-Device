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
static constexpr int UNKNOWN_CLASS = 5;

static constexpr int FFT_N_LOCAL = 4096;
static constexpr int SPEC_LEN = FFT_N_LOCAL / 2 + 1;

// segment kan variere i længde, men vi sætter et maksimum
static constexpr int MAX_SEGMENT_SAMPLES = 16000;   // ~1 sekund ved 16 kHz

// vinduer inde i segmentet
static constexpr int WINDOW_SIZE = 4096;
static constexpr int WINDOW_HOP = 2048;             // 50% overlap

// VAD thresholds
static constexpr float START_THRESHOLD = 0.03f;
static constexpr float STOP_THRESHOLD  = 0.015f;
static constexpr unsigned long SILENCE_TIMEOUT_MS = 250;

// cooldown efter detection
static constexpr unsigned long DETECTION_COOLDOWN_MS = 500;

// =========================
// LABELS
// =========================
static const char* CLASS_NAMES[N_CLASSES] = {
    "får", "ged", "hest", "laks", "ulv", "unknown"
};

// =========================
// BUFFERS
// =========================
static int16_t segmentBuffer[MAX_SEGMENT_SAMPLES];
static int segmentLength = 0;

// arbejdsbuffer til ét 4096-vindue
static float inferenceBuffer[FFT_N_LOCAL];

// feature-buffers
static float freqs[SPEC_LEN];
static float mag[SPEC_LEN];

// =========================
// STATE
// =========================
enum VadState {
    VAD_IDLE = 0,
    VAD_RECORDING
};

static uint8_t vadState = VAD_IDLE;
static unsigned long lastAboveThresholdMs = 0;
static unsigned long lastDetectionMs = 0;

// =========================
// HELPERS
// =========================
float rms_int16(const int16_t* x, size_t n) {
    if (n == 0) return 0.0f;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float v = static_cast<float>(x[i]) / 32768.0f;
        sum += v * v;
    }
    return std::sqrt(sum / n + 1e-12);
}

int predict_single_window_from_int16(const int16_t* audio, int n) {
    // copy + zero-pad/truncate til 4096 samples
    const int copy_len = (n < FFT_N_LOCAL) ? n : FFT_N_LOCAL;

    for (int i = 0; i < FFT_N_LOCAL; ++i) {
        inferenceBuffer[i] = 0.0f;
    }

    for (int i = 0; i < copy_len; ++i) {
        inferenceBuffer[i] = static_cast<float>(audio[i]) / 32768.0f;
    }

    // peak-normalisering så live data ligner træningsdata mere
    float peak = 1e-9f;
    for (int i = 0; i < copy_len; ++i) {
        float a = std::fabs(inferenceBuffer[i]);
        if (a > peak) peak = a;
    }
    if (peak > 1e-6f) {
        for (int i = 0; i < copy_len; ++i) {
            inferenceBuffer[i] /= peak;
        }
    }

    float rms = rms_energy_cpp(inferenceBuffer, FFT_N_LOCAL);
    float zcr = zero_crossing_rate_cpp(inferenceBuffer, FFT_N_LOCAL);

    magnitude_spectrum_cpp(inferenceBuffer, FFT_N_LOCAL, FS, freqs, mag, SPEC_LEN);

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

    return keyword_model_predict(features, 9);
}

int classify_segment_with_voting(const int16_t* audio, int n) {
    int votes[N_CLASSES] = {0};
    int numWindows = 0;

    if (n <= 0) {
        return UNKNOWN_CLASS;
    }

    // Hvis segmentet er kortere end ét vindue:
    if (n <= WINDOW_SIZE) {
        int pred = predict_single_window_from_int16(audio, n);
        if (pred >= 0 && pred < N_CLASSES) {
            votes[pred]++;
        }
        numWindows = 1;
    } else {
        // Kør flere 4096-vinduer med overlap
        for (int start = 0; start + WINDOW_SIZE <= n; start += WINDOW_HOP) {
            int pred = predict_single_window_from_int16(audio + start, WINDOW_SIZE);
            if (pred >= 0 && pred < N_CLASSES) {
                votes[pred]++;
            }
            numWindows++;
        }

        // Hvis der er en rest til sidst, som ikke blev dækket, så tag også sidste vindue
        int lastStart = n - WINDOW_SIZE;
        if (lastStart > 0 && (lastStart % WINDOW_HOP) != 0) {
            int pred = predict_single_window_from_int16(audio + lastStart, WINDOW_SIZE);
            if (pred >= 0 && pred < N_CLASSES) {
                votes[pred]++;
            }
            numWindows++;
        }
    }

    // Først: find bedste keyword-klasse (ignorér unknown)
    int bestKeywordClass = -1;
    int bestKeywordVotes = -1;
    for (int c = 0; c < N_CLASSES; ++c) {
        if (c == UNKNOWN_CLASS) continue;
        if (votes[c] > bestKeywordVotes) {
            bestKeywordVotes = votes[c];
            bestKeywordClass = c;
        }
    }

    // unknown-votes
    const int unknownVotes = votes[UNKNOWN_CLASS];

    // strategi:
    // hvis der er mindst én keyword-vote og keyword-votes >= unknown-votes, så vælg keyword
    // ellers vælg global vinder
    if (bestKeywordVotes > 0 && bestKeywordVotes >= unknownVotes) {
        Serial.printf("Voting result: keyword=%s votes=%d unknownVotes=%d windows=%d\r\n",
            CLASS_NAMES[bestKeywordClass], bestKeywordVotes, unknownVotes, numWindows);
        return bestKeywordClass;
    }

    // ellers vælg global vinder
    int bestClass = 0;
    for (int c = 1; c < N_CLASSES; ++c) {
        if (votes[c] > votes[bestClass]) {
            bestClass = c;
        }
    }

    Serial.printf("Voting result: best=%s votes=%d windows=%d\r\n",
        CLASS_NAMES[bestClass], votes[bestClass], numWindows);

    return bestClass;
}

void process_audio_chunk(const int16_t* src, size_t numSamples) {
    const float chunkRms = rms_int16(src, numSamples);
    const unsigned long now = millis();

    if (vadState == VAD_IDLE) {
        if (chunkRms >= START_THRESHOLD &&
            (now - lastDetectionMs) > DETECTION_COOLDOWN_MS) {

            vadState = VAD_RECORDING;
            segmentLength = 0;
            lastAboveThresholdMs = now;

            Serial.printf("VAD START rms=%.4f\r\n", chunkRms);
        }
    }

    if (vadState == VAD_RECORDING) {
        for (size_t i = 0; i < numSamples; ++i) {
            if (segmentLength < MAX_SEGMENT_SAMPLES) {
                segmentBuffer[segmentLength++] = src[i];
            }
        }

        if (chunkRms >= STOP_THRESHOLD) {
            lastAboveThresholdMs = now;
        }

        const bool silenceTooLong = (now - lastAboveThresholdMs) > SILENCE_TIMEOUT_MS;
        const bool bufferFull = segmentLength >= MAX_SEGMENT_SAMPLES;

        if (silenceTooLong || bufferFull) {
            Serial.printf("VAD STOP len=%d rms=%.4f\r\n", segmentLength, chunkRms);

            int pred = classify_segment_with_voting(segmentBuffer, segmentLength);

            if (pred >= 0 && pred < N_CLASSES) {
                Serial.printf("SEGMENT DETECTED -> pred=%d label=%s len=%d\r\n",
                    pred, CLASS_NAMES[pred], segmentLength);
            } else {
                Serial.printf("SEGMENT DETECTED -> pred=%d len=%d\r\n", pred, segmentLength);
            }

            lastDetectionMs = now;
            segmentLength = 0;
            vadState = VAD_IDLE;
        }
    }
}

// =========================
// SETUP / LOOP
// =========================
void setup() {
    Serial.begin(115200);
    waitFor(Serial.isConnected, 10000);
    delay(2000);

    Serial.println("Starting live KWS with VAD + voting...");

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