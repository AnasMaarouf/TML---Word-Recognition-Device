# TML Security Control Using Keyword Spotting on Particle Photon 2

## Overview

This project revolves around creating a TinyML solution on the Particle Photon2 board. Along with the board is a PDM microphone sensor. 

The overall idea for the project is to perform multiclass classification between different keywords. The list of labelled positive keywords is as follows:
- "Får"
- "Ged"
- "Hest"
- "Laks"
- "Ulv"

Also containing negative class labelled as unknown. 

This project divides the TinyML pipeline, by using both traditional machine learning and also a deep learning algorithm. 


## Repository Structure

```
TML---Word_recognition-Device/
├── Dataopsamling/
│   └── DAQ_KWS/
│       ├── lib/
│       │   └── Microphone_PDM/                     # PDM-microphone library, used in source
│       ├── server/
│       │   └── out/
│       │       ├── får_A_0.75m_stille.001.wav      # 30 second recordings (.WAV)
│       │       └── ...                             # More Recordings (.WAV)
│       └── src/
│           └── DAQ_KWS.cpp                         # Data aqqusition source code (.cpp)
│
├── ML model/
│   ├── Tidligere_arbejde/                   # Unstructured directory containing previous traditional ML work
│   └── TraditionelMLPipeline/               # Pipeline for traditionel ML, som bruges i rapporten, dog med ekstra funktionaliteter. Læs "TraditionelMLPipeline.ipynb" for Python kode, hvortil der er referencer til C++ kode i den samme rækkefølge, som pipelinen blev udført med. 
│       ├── 1b/                              # Step 1b of EMLearn's deployment strategy
│       ├── 1c/                              # Step 1c of EMLearn's deployment strategy
│       ├── 2/                               # Step 2 of EMLearn's deployment strategy
│       ├── feature_comparison_FFT4096_MultiWindow_Mel10/
│       │                                    # C++ feature extraction using 10 averaged Mel coefficients
│       ├── keywords/                        # Manually spliced 30-second labelled keyword recordings
│       ├── prepared_words/                  # Automatically spliced 1-second keyword clips used for training
│       ├── unknown/                         # Manually spliced 30-second unknown-class recordings
│       ├── unknown_balanced/                # Automatically spliced 1-second unknown clips used for training
│       ├── Test_Chjem1/                     # First test set recorded at Christian's apartment
│       ├── Test_Chjem2/                     # Second test set recorded at Christian's apartment
│       ├── test_ready_chjem1/               # Automatically spliced 1-second clips from the first test set
│       ├── test_ready_chjem2/               # Automatically spliced 1-second clips from the second test set
│       └── TraditionelMLPipeline.ipynb      # Main notebook for the traditional ML pipeline
│
├── DL model/
└── README.md
```

---

## Authors
**Christian Rex Rønfeldt Brandt Pilegaard, 6. semester - Diplomingeniør i Eleketronik studerende**

Tiny Machine Learning (ETTML) Projekt — Aarhus Universitet
