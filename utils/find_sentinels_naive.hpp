#ifndef FIND_SENTINELS_NAIVE_HPP
#define FIND_SENTINELS_NAIVE_HPP

#include "image.hpp"
#include <vector>

// Compare les bits LSB entre l'image originale et la porteuse
// et retourne les indices où les bits diffèrent
void find_sentinels_naive(const Image &carrier, const Image &original, std::vector<int> &sentinelIndices);

#endif // FIND_SENTINELS_NAIVE_HPP
