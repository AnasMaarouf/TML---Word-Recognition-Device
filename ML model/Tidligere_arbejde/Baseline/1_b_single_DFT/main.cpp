#include <stdio.h>
#include <stdint.h>

#include "audio_sample.h"
#include "keyword_model.h"
#include "features.h"

int main(void)
{
    const int n = audio_sample_length;
    const float fs = 16000.0f;
    const int spec_len = n / 2 + 1;

    float freqs[spec_len];
    float mag[spec_len];

    // Beregn simple features fra raw audio
    float rms = rms_energy_cpp(audio_sample, n);
    float zcr = zero_crossing_rate_cpp(audio_sample, n);

    magnitude_spectrum_cpp(audio_sample, n, fs, freqs, mag, spec_len);

    float centroid = spectral_centroid_cpp(freqs, mag, spec_len);
    float rolloff = spectral_rolloff_cpp(freqs, mag, spec_len, 0.85f);
    float dominant = dominant_frequency_cpp(freqs, mag, spec_len);
    float band0_500 = band_energy_cpp(freqs, mag, spec_len, 0.0f, 500.0f);
    float band500_1500 = band_energy_cpp(freqs, mag, spec_len, 500.0f, 1500.0f);
    float band1500_3000 = band_energy_cpp(freqs, mag, spec_len, 1500.0f, 3000.0f);
    float band3000_6000 = band_energy_cpp(freqs, mag, spec_len, 3000.0f, 6000.0f);

    // Feature order skal matche Python "simple_features"
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

    int out = keyword_model_predict(features, 9);

    printf("Predicted class id: %d\n", out);

    // valgfrit: print features også
    printf("rms           = %.8f\n", rms);
    printf("zcr           = %.8f\n", zcr);
    printf("centroid      = %.8f\n", centroid);
    printf("rolloff       = %.8f\n", rolloff);
    printf("dominant_freq = %.8f\n", dominant);
    printf("band_0_500    = %.8f\n", band0_500);
    printf("band_500_1500 = %.8f\n", band500_1500);
    printf("band_1500_3000= %.8f\n", band1500_3000);
    printf("band_3000_6000= %.8f\n", band3000_6000);

    return 0;
}