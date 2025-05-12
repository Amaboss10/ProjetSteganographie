#pragma once
#include "image.hpp"
#include <vector>
#include <string>

// Version avec détection via sentinelles
void naive_decrypt(const Image &carrier, const std::vector<int> &sentinels, const std::string &outputPath);