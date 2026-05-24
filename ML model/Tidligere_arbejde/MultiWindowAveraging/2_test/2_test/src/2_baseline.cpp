#include "Particle.h"
#include "Microphone_PDM.h"

#include "keyword_model_simple_fft4096.h"
#include "features.h"

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(SEMI_AUTOMATIC);

// SerialLogHandler logHandler;   // FJERN DENNE

static constexpr float FS = 16000.0f;
static constexpr int N_CLASSES = 6;

static constexpr int WINDOW_SIZE = 4096;
static constexpr int OVERLAP_SIZE = WINDOW_SIZE / 2;
static constexpr int HOP_SIZE = WINDOW_SIZE / 2;
static constexpr int SPEC_LEN = WINDOW_SIZE / 2 + 1;

static const char* CLASS_NAMES[N_CLASSES] = {
    "får", "ged", "hest", "laks", "ulv", "unknown"
};

static int16_t sampleBufferA[WINDOW_SIZE];
static int16_t sampleBufferB[WINDOW_SIZE];

static int16_t* fillBuffer = sampleBufferA;
static int16_t* processBuffer = nullptr;

static float inferenceFloatBuffer[WINDOW_SIZE];

static uint16_t fillIndex = 0;
static uint8_t processReady = 0;
static uint8_t appState = 0;

static float freqs[SPEC_LEN];
static float mag[SPEC_LEN];

enum AppState {
    STATE_INIT = 0,
    STATE_FILL_FIRST_WINDOW,
    STATE_RUN
};

void ingest_samples(const int16_t* src, size_t numSamples);
void run_inference_on_window(const int16_t* audio, int n);
void prepare_next_fill_buffer(int16_t* justCompletedBuffer);

void setup() {
    Serial.begin(115200);
    waitFor(Serial.isConnected, 10000);
    delay(2000);

    Serial.println("Starting live KWS with dual sample buffer...");

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

    fillBuffer = sampleBufferA;
    processBuffer = nullptr;
    fillIndex = 0;
    processReady = 0;
    appState = STATE_FILL_FIRST_WINDOW;
}

void loop() {
    if (appState == STATE_INIT) {
        delay(10);
        return;
    }

    Microphone_PDM::instance().noCopySamples([](void* pSamples, size_t numSamples) {
        const int16_t* s = static_cast<const int16_t*>(pSamples);
        ingest_samples(s, numSamples);
    });

    if (processReady && processBuffer != nullptr) {
        processReady = 0;
        run_inference_on_window(processBuffer, WINDOW_SIZE);
    }

    delay(1);
}

void ingest_samples(const int16_t* src, size_t numSamples) {
    for (size_t i = 0; i < numSamples; ++i) {
        if (fillIndex < WINDOW_SIZE) {
            fillBuffer[fillIndex++] = src[i];
        }

        if (fillIndex >= WINDOW_SIZE) {
            processBuffer = fillBuffer;
            processReady = 1;

            prepare_next_fill_buffer(fillBuffer);

            appState = STATE_RUN;
        }
    }
}

void prepare_next_fill_buffer(int16_t* justCompletedBuffer) {
    int16_t* nextBuffer = (justCompletedBuffer == sampleBufferA) ? sampleBufferB : sampleBufferA;

    for (int i = 0; i < OVERLAP_SIZE; ++i) {
        nextBuffer[i] = justCompletedBuffer[HOP_SIZE + i];
    }

    fillBuffer = nextBuffer;
    fillIndex = OVERLAP_SIZE;
}

void run_inference_on_window(const int16_t* audio, int n) {
    for (int i = 0; i < n; ++i) {
        inferenceFloatBuffer[i] = static_cast<float>(audio[i]) / 32768.0f;
    }

    float rms = rms_energy_cpp(inferenceFloatBuffer, n);
    float zcr = zero_crossing_rate_cpp(inferenceFloatBuffer, n);

    magnitude_spectrum_cpp(inferenceFloatBuffer, n, FS, freqs, mag, SPEC_LEN);

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
            "pred=%d label=%s rms=%.4f zcr=%.4f centroid=%.1f rolloff=%.1f dom=%.1f\r\n",
            pred,
            CLASS_NAMES[pred],
            rms,
            zcr,
            centroid,
            rolloff,
            dominant
        );
    } else {
        Serial.printf("pred=%d (invalid)\r\n", pred);
    }
}