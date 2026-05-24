#include <iostream>
#include <iomanip>

#include "audio_sample.h"
#include "features.h"

int main() {
    const float fs = 16000.0f;
    const int n = audio_sample_length;

    float features[SIMPLE_MEL_FEATURES_N];

    bool ok = extract_simple_mel_features_cpp(
        audio_sample,
        n,
        fs,
        features
    );

    if (!ok) {
        std::cout << "Feature extraction failed\n";
        return 1;
    }

    const char* names[SIMPLE_MEL_FEATURES_N] = {
        "rms",
        "zcr",
        "centroid",
        "rolloff",
        "dominant_freq",
        "band_0_500",
        "band_500_1500",
        "band_1500_3000",
        "band_3000_6000",

        "mel_1", "mel_2", "mel_3", "mel_4", "mel_5",
        "mel_6", "mel_7", "mel_8", "mel_9", "mel_10",
        "mel_11", "mel_12", "mel_13", "mel_14", "mel_15",
        "mel_16", "mel_17", "mel_18", "mel_19", "mel_20"
    };

    std::cout << std::fixed << std::setprecision(8);

    for (int i = 0; i < SIMPLE_MEL_FEATURES_N; ++i) {
        std::cout << std::setw(16) << names[i]
                  << " = "
                  << features[i]
                  << "\n";
    }

    return 0;
}