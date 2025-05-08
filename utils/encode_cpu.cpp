#include "encode_cpu.hpp"
#include <iostream>

void cpu_encode(Image &carrierImage, const unsigned char *hiddenData, int hiddenSize) {
    unsigned char *carrierData = carrierImage.data();
    int carrierSize = carrierImage.byteSize();

    int requiredBits = 8 + 32 + hiddenSize * 8; // start sentinel + size + data

    if (requiredBits > carrierSize) {
        std::cerr << "Erreur : image porteuse trop petite pour cacher les données." << std::endl;
        return;
    }

    int bitIndex = 0;

    // Sentinelle de début (8 bits à 1)
    for (int i = 0; i < 8; ++i, ++bitIndex) {
        carrierData[bitIndex] = (carrierData[bitIndex] & 0xFE) | 1;
    }

    // Encodage de la taille (32 bits)
    for (int i = 31; i >= 0; --i, ++bitIndex) {
        unsigned char bit = (hiddenSize >> i) & 1;
        carrierData[bitIndex] = (carrierData[bitIndex] & 0xFE) | bit;
    }

    // Encodage des données
    for (int i = 0; i < hiddenSize; ++i) {
        for (int b = 7; b >= 0; --b, ++bitIndex) {
            unsigned char bit = (hiddenData[i] >> b) & 1;
            carrierData[bitIndex] = (carrierData[bitIndex] & 0xFE) | bit;
        }
    }

    std::cout << "Encodage terminé : " << hiddenSize << " octets." << std::endl;
}
