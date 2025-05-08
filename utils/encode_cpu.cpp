#include "encode_cpu.hpp"
#include <iostream>

void cpu_encode(Image &carrierImage, const unsigned char *hiddenData, int hiddenSize) {
    unsigned char *carrierData = carrierImage.data();
    int carrierSize = carrierImage.byteSize();

    // Vérification : a-t-on assez de place pour cacher les données ?
    if (hiddenSize * 8 + 16 > carrierSize) {
        std::cerr << "Erreur : image porteuse trop petite pour cacher l'image." << std::endl;
        return;
    }

    // 1. Insertion de la sentinelle de début (ex : 8 bits à 1)
    for (int i = 0; i < 8; ++i) {
        carrierData[i] = (carrierData[i] & 0xFE) | 1;
    }

    // 2. Insertion des données cachées
    int bitIndex = 8; // on commence après la sentinelle de début
    for (int i = 0; i < hiddenSize; ++i) {
        for (int bit = 7; bit >= 0; --bit) {
            unsigned char bitToHide = (hiddenData[i] >> bit) & 1;
            carrierData[bitIndex] = (carrierData[bitIndex] & 0xFE) | bitToHide;
            ++bitIndex;
        }
    }

    // 3. Insertion de la sentinelle de fin (ex : 8 bits à 0)
    for (int i = 0; i < 8; ++i) {
        carrierData[bitIndex + i] = (carrierData[bitIndex + i] & 0xFE);
    }
}
