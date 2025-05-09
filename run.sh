#!/bin/bash

set -e

# Compilation
cmake -S . -B build
cmake --build build

# Lancement selon les arguments
if [ "$1" == "decrypt" ]; then
    if [ "$2" == "gpu" ]; then
        ./build/ProjetSteganographieBase decrypt gpu
    else
        ./build/ProjetSteganographieBase decrypt
    fi
elif [ "$1" == "gpu" ]; then
    ./build/ProjetSteganographieBase gpu
else
    ./build/ProjetSteganographieBase
fi
