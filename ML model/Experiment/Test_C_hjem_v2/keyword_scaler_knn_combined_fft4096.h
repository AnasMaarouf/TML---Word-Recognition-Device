#ifndef KEYWORD_SCALER_KNN_COMBINED_FFT4096_H
#define KEYWORD_SCALER_KNN_COMBINED_FFT4096_H

#include <stdint.h>
#include <math.h>

#define KEYWORD_N_FEATURES 42
#define KNN_Q_SCALE 1024.0f

static const float keyword_scaler_mean[KEYWORD_N_FEATURES] = {
    1.478624195e-01f,
    1.701676100e-01f,
    2.315734619e+03f,
    5.208802246e+03f,
    2.772265625e+02f,
    2.428710352e+04f,
    2.433332520e+03f,
    9.167346191e+02f,
    1.133896851e+03f,
    -1.122140427e+02f,
    1.383106136e+01f,
    9.705277443e+00f,
    4.631041050e+00f,
    7.268828154e-01f,
    -2.566926241e+00f,
    -1.363465786e+00f,
    -2.462205648e+00f,
    -1.221255302e+00f,
    -5.948293209e-01f,
    -8.079608679e-01f,
    -4.339009225e-01f,
    -1.436175257e-01f,
    8.742024302e-01f,
    6.231719851e-01f,
    6.200560927e-01f,
    1.801177710e-01f,
    1.008712947e-01f,
    9.502756596e-02f,
    3.557291627e-02f,
    3.447700292e-02f,
    2.938274108e-02f,
    3.034148365e-02f,
    2.428699657e-02f,
    2.511426248e-02f,
    2.510493435e-02f,
    2.570093796e-02f,
    2.926595882e-02f,
    2.537887543e-02f,
    2.487784810e-02f,
    3.003139608e-02f,
    3.815796599e-02f,
    4.261850566e-02f,
};

static const float keyword_scaler_scale[KEYWORD_N_FEATURES] = {
    4.953588173e-02f,
    6.033879146e-02f,
    6.855454712e+02f,
    1.234926514e+03f,
    3.904981995e+02f,
    6.498256250e+04f,
    1.087941211e+04f,
    2.947548340e+03f,
    3.063264893e+03f,
    3.846169281e+01f,
    8.222779274e+00f,
    4.628126144e+00f,
    5.105818748e+00f,
    4.032960892e+00f,
    3.349559784e+00f,
    2.919558048e+00f,
    2.713957310e+00f,
    2.461195469e+00f,
    2.369656086e+00f,
    1.891254783e+00f,
    1.612279534e+00f,
    1.474526286e+00f,
    2.972876072e+00f,
    2.046059847e+00f,
    2.191618681e+00f,
    8.614605069e-01f,
    4.011816382e-01f,
    7.389024496e-01f,
    1.689441055e-01f,
    2.214833349e-01f,
    1.230373979e-01f,
    1.251590103e-01f,
    7.927132398e-02f,
    1.169549003e-01f,
    7.695315778e-02f,
    7.286997139e-02f,
    9.463886917e-02f,
    6.998982280e-02f,
    7.335748523e-02f,
    8.634831756e-02f,
    1.084754765e-01f,
    1.154370010e-01f,
};

static inline void keyword_scale_features(float *features) {
    for (int i = 0; i < KEYWORD_N_FEATURES; i++) {
        features[i] = (features[i] - keyword_scaler_mean[i]) / keyword_scaler_scale[i];
    }
}

static inline void keyword_quantize_features(const float *features, int16_t *features_i16) {
    for (int i = 0; i < KEYWORD_N_FEATURES; i++) {
        float v = features[i] * KNN_Q_SCALE;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        features_i16[i] = (int16_t)lroundf(v);
    }
}

#endif
