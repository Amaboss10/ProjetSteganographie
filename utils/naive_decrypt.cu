#include "naive_decrypt.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void extractBitsKernel(const unsigned char *carrier, const int *sentinels,
                                  unsigned char *output, int numBits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBits) {
        int byteIdx = i / 8;
        int bitPos  = 7 - (i % 8);
        int index   = sentinels[i];
        unsigned char bit = (carrier[index] & 1) << bitPos;

        // Écriture directe (sans atomic)
        output[byteIdx] |= bit;
    }
}

void naive_decrypt(const Image &carrier, const std::vector<int> &sentinels, const std::string &outputPath) {
    const unsigned char *carrierData = carrier.data();
    int totalBits = static_cast<int>(sentinels.size());
    int totalBytes = totalBits / 8;

    // === Allocation mémoire GPU ===
    unsigned char *d_carrier = nullptr;
    int *d_sentinels = nullptr;
    unsigned char *d_output = nullptr;

    cudaMalloc(&d_carrier, carrier.byteSize());
    cudaMalloc(&d_sentinels, sizeof(int) * totalBits);
    cudaMalloc(&d_output, sizeof(unsigned char) * totalBytes);
    cudaMemset(d_output, 0, totalBytes);

    cudaMemcpy(d_carrier, carrierData, carrier.byteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sentinels, sentinels.data(), sizeof(int) * totalBits, cudaMemcpyHostToDevice);

    // === Lancer le kernel ===
    int threads = 256;
    int blocks = (totalBits + threads - 1) / threads;
    extractBitsKernel<<<blocks, threads>>>(d_carrier, d_sentinels, d_output, totalBits);
    cudaDeviceSynchronize();

    // === Copier le résultat côté CPU ===
    unsigned char *output = new unsigned char[totalBytes];
    cudaMemcpy(output, d_output, totalBytes, cudaMemcpyDeviceToHost);

    // === Afficher les premiers octets extraits ===
    std::cout << "Premiers octets extraits (hex) : ";
    for (int i = 0; i < 16 && i < totalBytes; ++i)
        std::cout << std::hex << std::uppercase << (int)output[i] << " ";
    std::cout << std::dec << std::endl;

    // === Reconstruire l'image à partir des octets extraits ===
    Image decoded;
    if (decoded.createFromRawImage(output, totalBytes)) {
        decoded.save(outputPath);
        std::cout << "Image décodée par CUDA enregistrée dans : " << outputPath << std::endl;
    } else {
        std::cerr << "Erreur : impossible de reconstruire l'image décodée (format inconnu ?)." << std::endl;
    }

    // === Nettoyage mémoire ===
    delete[] output;
    cudaFree(d_carrier);
    cudaFree(d_sentinels);
    cudaFree(d_output);
}
