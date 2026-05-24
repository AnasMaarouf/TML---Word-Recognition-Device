#include <iostream>
#include <iomanip>

#include "audio_sample.h"
#include "features.h"

int main() {
    const float fs = 16000.0f;
    const int n = audio_sample_length;

    float features[MEL_FEATURES_N];

    bool ok = extract_mel_features_cpp(
        audio_sample,
        n,
        fs,
        features
    );

    if (!ok) {
        std::cout << "Feature extraction failed\n";
        return 1;
    }

    const char* names[MEL_FEATURES_N] = {
        "mel_1",
        "mel_2",
        "mel_3",
        "mel_4",
        "mel_5",
        "mel_6",
        "mel_7",
        "mel_8",
        "mel_9",
        "mel_10"
    };

    std::cout << std::fixed << std::setprecision(8);

    for (int i = 0; i < MEL_FEATURES_N; ++i) {
        std::cout << std::setw(10) << names[i]
                  << " = "
                  << features[i]
                  << "\n";
    }

    return 0;
}