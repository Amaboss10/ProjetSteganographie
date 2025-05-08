#include "encode_naive.hpp"
#include <cuda_runtime.h>
#include <iostream>

// Kernel CUDA : chaque thread encode 1 bit dans un octet de l'image
__global__ void encodeKernel(unsigned char *carrier, const unsigned char *data, int hiddenSize, int carrierSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Décalage après la sentinelle et la taille
    int bitOffset = 8 + 32;

    if (i < hiddenSize * 8) {
        int byteIndex = i / 8;
        int bitInByte = 7 - (i % 8);
        unsigned char bit = (data[byteIndex] >> bitInByte) & 1;

        int carrierIndex = bitOffset + i;
        if (carrierIndex < carrierSize) {
            carrier[carrierIndex] = (carrier[carrierIndex] & 0xFE) | bit;
        }
    }
}

// Fonction appelée depuis le code C++ pour lancer le kernel
void encode_naive(Image &carrierImage, const unsigned char *hiddenData, int hiddenSize) {
    unsigned char *carrierData = carrierImage.data();
    int carrierSize = carrierImage.byteSize();

    int requiredBits = 8 + 32 + hiddenSize * 8;
    if (requiredBits > carrierSize) {
        std::cerr << "Erreur : image porteuse trop petite pour les données." << std::endl;
        return;
    }

    // Encode la sentinelle de début (8 bits à 1)
    for (int i = 0; i < 8; ++i) {
        carrierData[i] = (carrierData[i] & 0xFE) | 1;
    }

    // Encode la taille sur 32 bits
    for (int i = 31; i >= 0; --i) {
        unsigned char bit = (hiddenSize >> i) & 1;
        carrierData[8 + (31 - i)] = (carrierData[8 + (31 - i)] & 0xFE) | bit;
    }

    // Allocation GPU
    unsigned char *d_carrier, *d_data;
    cudaMalloc(&d_carrier, carrierSize);
    cudaMalloc(&d_data, hiddenSize);

    cudaMemcpy(d_carrier, carrierData, carrierSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, hiddenData, hiddenSize, cudaMemcpyHostToDevice);

    // Lancement du kernel
    int threadsPerBlock = 256;
    int totalBits = hiddenSize * 8;
    int blocks = (totalBits + threadsPerBlock - 1) / threadsPerBlock;
    encodeKernel<<<blocks, threadsPerBlock>>>(d_carrier, d_data, hiddenSize, carrierSize);

    cudaMemcpy(carrierData, d_carrier, carrierSize, cudaMemcpyDeviceToHost);

    // Libération GPU
    cudaFree(d_carrier);
    cudaFree(d_data);

    std::cout << "Encodage CUDA terminé : " << hiddenSize << " octets." << std::endl;
}
