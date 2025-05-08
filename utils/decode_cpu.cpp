#include "decode_cpu.hpp"
#include <iostream>
#include <vector>
#include <fstream>

void cpu_decode(const Image &carrierImage, const std::string &outputPath) {
    const unsigned char *carrierData = carrierImage.data();
    int carrierSize = carrierImage.byteSize();

    // 1. Rechercher la sentinelle de début (8 bits à 1)
    int start = -1;
    for (int i = 0; i <= carrierSize - 40; ++i) {
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

    // 2. Lire la taille sur 32 bits
    int size = 0;
    for (int i = 0; i < 32; ++i) {
        size = (size << 1) | (carrierData[start + i] & 0x01);
    }

    std::cout << "Taille attendue : " << size << " octets." << std::endl;

    // 3. Extraire les bits des données
    std::vector<unsigned char> bits;
    int bitStart = start + 32;
    for (int i = 0; i < size * 8 && (bitStart + i) < carrierSize; ++i) {
        bits.push_back(carrierData[bitStart + i] & 0x01);
    }

    // 4. Reconstituer les octets
    std::vector<unsigned char> hiddenData;
    for (size_t i = 0; i + 7 < bits.size(); i += 8) {
        unsigned char byte = 0;
        for (int j = 0; j < 8; ++j) {
            byte = (byte << 1) | bits[i + j];
        }
        hiddenData.push_back(byte);
    }

    std::cout << "Octets extraits : " << hiddenData.size() << std::endl;

    // Sauvegarde brute pour inspection
    std::ofstream out("images/debug_output.raw", std::ios::binary);
    out.write(reinterpret_cast<const char*>(hiddenData.data()), hiddenData.size());
    out.close();

    // 5. Création de l'image
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
