#ifndef NAIVE_DECRYPT_HPP
#define NAIVE_DECRYPT_HPP

#include "image.hpp"
#include <vector>
#include <string>

// Décode l'image cachée à partir des indices de sentinelles et la sauvegarde
void naive_decrypt(const Image &carrier, const std::vector<int> &sentinels, const std::string &outputPath);

#endif // NAIVE_DECRYPT_HPP
