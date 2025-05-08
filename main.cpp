#include <iostream>

#include "utils/image.hpp"
#include "utils/raw_image_loader.hpp"
#include "utils/encode_cpu.hpp" 
#include "utils/decode_cpu.hpp"


void invert( Image & image )
{
    unsigned char * imageData = image.data();
    for ( int i = 0; i < image.byteSize(); ++i )
    {
        imageData[ i ] = 255 - imageData[ i ];
    }
}

int main()
{
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

    // Loading raw image (it contains all the png stuff) and saving it
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

    // === Encodage LSB CPU ===
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

    cpu_encode(carrier, hiddenData, hiddenSize);

    if (!carrier.save("./images/boule_encoded.png"))
    {
        std::cerr << "Error: Could not save encoded image!" << std::endl;
        delete[] hiddenData;
        return -1;
    }

    delete[] hiddenData;

    return 0;
}
