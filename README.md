# TML---Word-Recognition-Device

**TinyML project**

This project revolves around creating a TinyML solution on the Particle Photon2 board. Along with the board is two sensors:

    A PDM microphone
    Temperature Sensor, perhaps LM75?

The overall idea for the project is:

    Vi vil bruge multiclass classification til at kunne klassificere mellem forskellige keyword/key sentences, såsom “Hey”, “Start”, “Stop”, “*støj*” 
        Vi bør være opmærksomme på at vi bør vælge ord, som ikke indeholder plosiver, såsom T, P, osv. Det vil fucke mikrofon målingen op
