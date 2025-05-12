#include "naive_decrypt.hpp"
#include "find_sentinels_naive.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Kernel pour extraire les bits LSB à partir des indices sentinelles
__global__ void extractBitsKernel(const unsigned char* carrier, const int* sentinels, int numBits, unsigned char* bitsOut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBits) {
        int pos = sentinels[idx];
        bitsOut[idx] = carrier[pos] & 1;
    }
}

void naive_decrypt(const Image& encoded, const Image& original, const std::string& outputPath) {
    const unsigned char* carrierData = encoded.data();
    const unsigned char* originalData = original.data();
    int size = encoded.byteSize();

    std::vector<int> sentinels;
    find_sentinels_naive(encoded, original, sentinels); // GPU : compare les LSB et note les différences

    int numBits = sentinels.size();

    // Allocation mémoire sur GPU
    unsigned char* d_carrier = nullptr;
    int* d_sentinels = nullptr;
    unsigned char* d_bits = nullptr;
    cudaMalloc(&d_carrier, size);
    cudaMalloc(&d_sentinels, sizeof(int) * numBits);
    cudaMalloc(&d_bits, sizeof(unsigned char) * numBits);

    cudaMemcpy(d_carrier, carrierData, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sentinels, sentinels.data(), sizeof(int) * numBits, cudaMemcpyHostToDevice);

    // Lancer le kernel
    int threads = 256;
    int blocks = (numBits + threads - 1) / threads;
    extractBitsKernel<<<blocks, threads>>>(d_carrier, d_sentinels, numBits, d_bits);

    // Récupérer les bits extraits
    std::vector<unsigned char> bits(numBits);
    cudaMemcpy(bits.data(), d_bits, sizeof(unsigned char) * numBits, cudaMemcpyDeviceToHost);

    // Reconstruction des octets
    std::vector<unsigned char> hiddenData;
    for (size_t i = 0; i + 7 < bits.size(); i += 8) {
        unsigned char byte = 0;
        for (int j = 0; j < 8; ++j) {
            byte = (byte << 1) | bits[i + j];
        }
        hiddenData.push_back(byte);
    }

    std::cout << "Premiers octets extraits (hex) : ";
    for (int i = 0; i < 16 && i < hiddenData.size(); ++i)
        std::cout << std::hex << (int)hiddenData[i] << " ";
    std::cout << std::dec << std::endl;

    // Reconstruction de l'image depuis les octets
    Image extracted;
    if (!extracted.createFromRawImage(hiddenData.data(), hiddenData.size())) {
        std::cerr << "Erreur : impossible de reconstruire l'image décodée (format inconnu ?)." << std::endl;
    } else {
        extracted.save(outputPath);
        std::cout << "Image cachée extraite avec succès dans : " << outputPath << std::endl;
    }

    // Libération mémoire GPU
    cudaFree(d_carrier);
    cudaFree(d_sentinels);
    cudaFree(d_bits);
}
