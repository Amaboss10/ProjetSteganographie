#include "decode_cpu.hpp"
#include <iostream>
#include <vector>

void cpu_decode(const Image &carrierImage, const std::string &outputPath) {
    const unsigned char *carrierData = carrierImage.data();
    int carrierSize = carrierImage.byteSize();

    // 1. Rechercher la sentinelle de début (8 bits à 1)
    int start = -1;
    for (int i = 0; i <= carrierSize - 8; ++i) {
        bool isStart = true;
        for (int j = 0; j < 8; ++j) {
            if ((carrierData[i + j] & 0x01) != 1) {
                isStart = false;
                break;
            }
        }
        if (isStart) {
            start = i + 8;
            break;
        }
    }
    if (start == -1) {
        std::cerr << "Erreur : sentinelle de début non trouvée." << std::endl;
        return;
    }

    // 2. Extraire les bits LSB jusqu’à la sentinelle de fin (8 bits à 0)
    std::vector<unsigned char> bits;
    for (int i = start; i < carrierSize - 8; ++i) {
        bool isEnd = true;
        for (int j = 0; j < 8; ++j) {
            if ((carrierData[i + j] & 0x01) != 0) {
                isEnd = false;
                break;
            }
        }
        if (isEnd)
            break;

        bits.push_back(carrierData[i] & 0x01);
    }

    // 3. Reconstituer les octets
    std::vector<unsigned char> hiddenData;
    for (size_t i = 0; i + 7 < bits.size(); i += 8) {
        unsigned char byte = 0;
        for (int j = 0; j < 8; ++j) {
            byte = (byte << 1) | bits[i + j];
        }
        hiddenData.push_back(byte);
    }

    // 4. Créer et sauvegarder l’image extraite
    Image extracted;
    if (!extracted.createFromRawImage(hiddenData.data(), hiddenData.size())) {
        std::cerr << "Erreur : création de l'image extraite échouée." << std::endl;
        return;
    }

    if (!extracted.save(outputPath)) {
        std::cerr << "Erreur : sauvegarde de l'image extraite échouée." << std::endl;
    } else {
        std::cout << "Image cachée extraite avec succès dans : " << outputPath << std::endl;
    }
}
