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
│   ├── DAQ_KWS/                    
        ├── lib/
            └── Microphone_PDM/             # PDM-microphone library, used in source
        ├── server/
            └── out/
                └── får_A_0.75m_stille.001  # 30 second recordings (.WAV)
                └── ...                     # More Recordings (.WAV)
        └── src/
            └── DAQ_KWS.cpp                 # Data aqqusition source code (.cpp)
├── ML model/
│   ├── Tidligere_arbejde/                  # Rodet directory, som indeholder alt tidligere arbejde med traditionel ML
    └── TraditionelMLPipeline/              # Pipeline for traditionel ML, som bruges i rapporten, dog med ekstra funktionaliteter. Læs "TraditionelMLPipeline.ipynb" for Python kode, hvortil der er referencer til C++ kode i den samme rækkefølge, som pipelinen blev udført med. 
        ├── 1b/                             # Directory for step "1b", mht. EMLearn's deployment strategi. 
        ├── 1c/                             # Directory for step "1c", mht. EMLearn's deployment strategi. 
        ├── 2/                              # Directory for step "2", mht. EMLearn's deployment strategi. 
        ├── feature_comparison_FFT4096_MultiWindow_Mel10/                             # Directory for feature extraction af 10 gennemsnitlige mel-koefficienter i C++.
        ├── keywords/                       # Directory containing 30 second audio clips of labelled keywords, after manually splicing the raw clips to avoid wrong automatic splicing.
        ├── prepared_words/                 # Directory containing automated spliced 1 second audio clips of labelled keywords. This is used for training the model.
        ├── unknown/                        # Directory containing 30 second audio clips of unknown labelled keywords, after manually splicing the raw clips to avoid wrong automatic splicing.
        ├── unknown_balanced/               # Directory containing automated spliced 1 second audio clips of labelled unknown keywords. This is used for training the model.
        ├── Test_Chjem1/                    # Directory containing 30 second audio clips of labelled keywords from test data at Christians apartment, resembling the "bachelorlokale"-location, where the training data was recorded. This is the first set, to differentiate from date to date variations from the training set.
        ├── Test_Chjem2/                    # Directory containing 30 second audio clips of labelled keywords from test data at Christians apartment, resembling the "bachelorlokale"-location, where the training data was recorded. This is the second set, to differentiate from date to date variations from the training set.
        ├── test_ready_chjem1/                 # Directory containing automated spliced 1 second audio clips of labelled keywords. This is used for evaluating the model, based of the first test set.
        ├── test_ready_chjem2/                 # Directory containing automated spliced 1 second audio clips of labelled keywords. This is used for evaluating the model, based of the second test set.
        
        
├── DL model/
└── README.md
```

---

## Authors
**Christian Rex Rønfeldt Brandt Pilegaard, 6. semester - Diplomingeniør i Eleketronik studerende**

Tiny Machine Learning (ETTML) Projekt — Aarhus Universitet
