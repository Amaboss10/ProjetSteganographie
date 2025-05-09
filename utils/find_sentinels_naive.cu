#include "find_sentinels_naive.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void sentinelKernel(const unsigned char *carrier, const unsigned char *original,
                               int *sentinels, int *counter, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        unsigned char lsb_c = carrier[i] & 1;
        unsigned char lsb_o = original[i] & 1;
        if (lsb_c != lsb_o) {
            int index = atomicAdd(counter, 1);
            sentinels[index] = i;
        }
    }
}

void find_sentinels_naive(const Image &carrier, const Image &original, std::vector<int> &sentinelIndices) {
    const unsigned char *carrierData = carrier.data();
    const unsigned char *originalData = original.data();
    int size = carrier.byteSize();

    // Allocation GPU
    unsigned char *d_carrier, *d_original;
    int *d_sentinels, *d_counter;

    cudaMalloc(&d_carrier, size);
    cudaMalloc(&d_original, size);
    cudaMalloc(&d_sentinels, sizeof(int) * size); 
    cudaMalloc(&d_counter, sizeof(int));

    cudaMemcpy(d_carrier, carrierData, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_original, originalData, size, cudaMemcpyHostToDevice);
    cudaMemset(d_counter, 0, sizeof(int));

    // Lancement du kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sentinelKernel<<<blocks, threads>>>(d_carrier, d_original, d_sentinels, d_counter, size);

    // Récupération
    int count = 0;
    cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    sentinelIndices.resize(count);
    cudaMemcpy(sentinelIndices.data(), d_sentinels, sizeof(int) * count, cudaMemcpyDeviceToHost);

    // Nettoyage
    cudaFree(d_carrier);
    cudaFree(d_original);
    cudaFree(d_sentinels);
    cudaFree(d_counter);

    std::cout << "Sentinelles trouvées : " << count << " positions différentes." << std::endl;
}
