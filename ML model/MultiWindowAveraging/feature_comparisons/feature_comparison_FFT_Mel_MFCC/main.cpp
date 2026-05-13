#include <iostream>
#include <iomanip>
#include <vector>

#include "audio_sample.h"
#include "features.h"

int main() {
    const int n = audio_sample_length;
    const float fs = 16000.0f;

    std::vector<float> mel(20);
    std::vector<float> mfcc(13);

    bool ok_mel = mel_spectrum_cpp(audio_sample, n, fs, mel.data(), 20);
    bool ok_mfcc = mfcc_features_cpp(audio_sample, n, fs, mfcc.data(), 13, 20);

    if (!ok_mel || !ok_mfcc) {
        std::cerr << "Mel/MFCC computation failed\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(8);

    std::cout << "Mel features:\n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "mel_" << i + 1 << " = " << mel[i] << "\n";
    }

    std::cout << "\nMFCC features:\n";
    for (int i = 0; i < 13; ++i) {
        std::cout << "mfcc_" << i + 1 << " = " << mfcc[i] << "\n";
    }

    return 0;
}