#include <iostream>
#include <string> 
#include <chrono>
#include <vector>

#include "utils/image.hpp"
#include "utils/raw_image_loader.hpp"
#include "utils/encode_cpu.hpp"
#include "utils/decode_cpu.hpp"
#include "utils/encode_naive.hpp"
#include "utils/find_sentinels_naive.hpp"

void invert( Image & image )
{
    unsigned char * imageData = image.data();
    for ( int i = 0; i < image.byteSize(); ++i )
    {
        imageData[ i ] = 255 - imageData[ i ];
    }
}

int main(int argc, char** argv)
{
    // Mode décodage uniquement
    if (argc > 1 && std::string(argv[1]) == "decrypt")
    {
        std::string inputEncoded = (argc > 2 && std::string(argv[2]) == "gpu")
                                   ? "./images/oizo_encoded_gpu.png"
                                   : "./images/oizo_encoded.png";

        std::string outputDecoded = (argc > 2 && std::string(argv[2]) == "gpu")
                                   ? "./images/oizo_decoded_gpu.png"
                                   : "./images/oizo_decoded.png";

        // === Décodage LSB CPU ===
        Image encodedImage;
        if (!encodedImage.load(inputEncoded))
        {
            std::cerr << "Error: Could not load encoded image!" << std::endl;
            return -1;
        }

        cpu_decode(encodedImage, outputDecoded);
        return 0;
    }

    // Loading and saving images
    Image image;
    if ( !image.load( "./images/agile.png" ) )
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    invert( image );

    if ( !image.save( "./images/agile_inverted.png" ) )
    {
        std::cerr << "Error: Could not save inverted image!" << std::endl;
        return -1;
    }

    // Loading raw image and saving it
    int             rawByteSize = 0;
    unsigned char * rawData     = loadRawImage( "./images/malice.png", rawByteSize );
    if ( !rawData )
    {
        std::cerr << "Error: Could not load raw image!" << std::endl;
        return -1;
    }

    Image rawImage;
    if ( !rawImage.createFromRawImage( rawData, rawByteSize ) )
    {
        std::cerr << "Error: Could not create image from raw data!" << std::endl;
        delete[] rawData;
        return -1;
    }
    delete[] rawData;

    invert( rawImage );

    if ( !rawImage.save( "./images/malice_inverted.png" ) )
    {
        std::cerr << "Error: Could not save inverted image!" << std::endl;
        return -1;
    }

    // === Encodage LSB (choix CPU / CUDA) ===
    Image carrier;
    if (!carrier.load("./images/oizo.png"))
    {
        std::cerr << "Error: Could not load carrier image!" << std::endl;
        return -1;
    }

    int hiddenSize = 0;
    unsigned char *hiddenData = loadRawImage("./images/malice.png", hiddenSize);
    if (!hiddenData)
    {
        std::cerr << "Error: Could not load hidden image!" << std::endl;
        return -1;
    }

    std::string outputPath;
    if (argc > 1 && std::string(argv[1]) == "gpu") {
        outputPath = "./images/oizo_encoded_gpu.png";

        auto start = std::chrono::high_resolution_clock::now();
        encode_naive(carrier, hiddenData, hiddenSize);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Temps CUDA (encode_naive) : " << duration.count() << " ms" << std::endl;

    } else {
        outputPath = "./images/oizo_encoded.png";

        auto start = std::chrono::high_resolution_clock::now();
        cpu_encode(carrier, hiddenData, hiddenSize);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Temps CPU (cpu_encode) : " << duration.count() << " ms" << std::endl;
    }

    if (!carrier.save(outputPath))
    {
        std::cerr << "Error: Could not save encoded image!" << std::endl;
        delete[] hiddenData;
        return -1;
    }

    delete[] hiddenData;

    // === Analyse des sentinelles CUDA ===
    std::vector<int> sentinels;
    Image original;
    if (!original.load("./images/oizo.png")) {
        std::cerr << "Error: Could not load original image for comparison!" << std::endl;
        return -1;
    }

    find_sentinels_naive(carrier, original, sentinels);
    std::cout << "Premiers indices modifiés : ";
    for (size_t i = 0; i < std::min<size_t>(10, sentinels.size()); ++i) {
        std::cout << sentinels[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
