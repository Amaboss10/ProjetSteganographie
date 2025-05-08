#pragma once

#include "image.hpp"
#include "raw_image_loader.hpp"

void cpu_encode(Image &carrierImage, const unsigned char *hiddenData, int hiddenSize);
