#include <iostream>
#include <string>
#include <chrono> // Pour mesurer les temps d'exécution

// Inclusions des modules du projet
#include "utils/image.hpp"               // Classe Image (chargement, sauvegarde, manipulation)
#include "utils/raw_image_loader.hpp"    // Chargement de l’image cachée au format brut (RAW)
#include "utils/encode_cpu.hpp"          // Encodage LSB en CPU
#include "utils/decode_cpu.hpp"          // Décodage LSB en CPU
#include "utils/encode_naive.hpp"        // Encodage LSB en GPU (version naïve CUDA)
#include "utils/naive_decrypt.hpp" 


int main(int argc, char** argv)
{
    // Affichage des arguments de la ligne de commande (utile pour le debug)
    std::cout << "Arguments (" << argc << "): ";
    for (int i = 0; i < argc; ++i) std::cout << argv[i] << " ";
    std::cout << std::endl;

    // === Mode décodage ===
    if (argc >= 2 && std::string(argv[1]) == "decrypt") {
        std::string encodedPath = "./images/encoded/rat_encoded.png";
        std::string outputPath  = "./images/decoded/rat_decoded.png";

        if (argc >= 3 && std::string(argv[2]) == "gpu") {
            encodedPath = "./images/encoded/rat_encoded_gpu.png";
            outputPath  = "./images/decoded/rat_decoded_gpu.png";
        }

        Image encodedImage;
        if (!encodedImage.load(encodedPath)) {
            std::cerr << "Erreur : impossible de charger l'image encodée : " << encodedPath << std::endl;
            return -1;
        }

        // Mesure du temps pour le décodage
        auto start = std::chrono::high_resolution_clock::now();
        cpu_decode(encodedImage, outputPath);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Temps CPU : "
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << " ms" << std::endl;

        return 0;
    }

    // === Mode encodage ===
    Image carrier;
    if (!carrier.load("./images/rat.png")) {
        std::cerr << "Erreur : impossible de charger l'image porteuse ./images/rat.png" << std::endl;
        return -1;
    }

    int hiddenSize = 0;
    unsigned char* hiddenData = loadRawImage("./images/soleil_nuit.png", hiddenSize);
    if (!hiddenData) {
        std::cerr << "Erreur : impossible de charger l'image cachée ./images/soleil_nuit.png" << std::endl;
        return -1;
    }

    std::string encodedPath = "./images/encoded/rat_encoded.png";

    if (argc >= 2 && std::string(argv[1]) == "gpu") {
        encodedPath = "./images/encoded/rat_encoded_gpu.png";

        auto start = std::chrono::high_resolution_clock::now();
        encode_naive(carrier, hiddenData, hiddenSize);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Encodage CUDA terminé : " << hiddenSize << " octets." << std::endl;
        std::cout << "Temps CUDA (encode_naive) : "
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << " ms" << std::endl;
    } else {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_encode(carrier, hiddenData, hiddenSize);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Encodage terminé : " << hiddenSize << " octets." << std::endl;
        std::cout << "Temps CPU : "
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << " ms" << std::endl;
    }

    if (!carrier.save(encodedPath)) {
        std::cerr << "Erreur : impossible de sauvegarder l'image encodée." << std::endl;
        delete[] hiddenData;
        return -1;
    }
  
    delete[] hiddenData;
    return 0;
}