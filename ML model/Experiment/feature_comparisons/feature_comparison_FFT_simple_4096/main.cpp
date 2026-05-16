#include <iostream>
#include <iomanip>
#include <vector>

#include "audio_sample.h"
#include "features.h"

static constexpr int FFT_N = 4096;

int main() {
    const int n = audio_sample_length;
    const float fs = 16000.0f;
    const int spec_len = FFT_N / 2 + 1;

    std::vector<float> freqs(spec_len);
    std::vector<float> mag(spec_len);

    float rms = rms_energy_cpp(audio_sample, n);
    float zcr = zero_crossing_rate_cpp(audio_sample, n);

    magnitude_spectrum_cpp(audio_sample, n, fs, freqs.data(), mag.data(), spec_len);

    float centroid = spectral_centroid_cpp(freqs.data(), mag.data(), spec_len);
    float rolloff = spectral_rolloff_cpp(freqs.data(), mag.data(), spec_len, 0.85f);
    float dominant = dominant_frequency_cpp(freqs.data(), mag.data(), spec_len);
    float band0_500 = band_energy_cpp(freqs.data(), mag.data(), spec_len, 0.0f, 500.0f);
    float band500_1500 = band_energy_cpp(freqs.data(), mag.data(), spec_len, 500.0f, 1500.0f);
    float band1500_3000 = band_energy_cpp(freqs.data(), mag.data(), spec_len, 1500.0f, 3000.0f);
    float band3000_6000 = band_energy_cpp(freqs.data(), mag.data(), spec_len, 3000.0f, 6000.0f);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "rms           = " << rms << "\n";
    std::cout << "zcr           = " << zcr << "\n";
    std::cout << "centroid      = " << centroid << "\n";
    std::cout << "rolloff       = " << rolloff << "\n";
    std::cout << "dominant_freq = " << dominant << "\n";
    std::cout << "band_0_500    = " << band0_500 << "\n";
    std::cout << "band_500_1500 = " << band500_1500 << "\n";
    std::cout << "band_1500_3000= " << band1500_3000 << "\n";
    std::cout << "band_3000_6000= " << band3000_6000 << "\n";

    return 0;
}