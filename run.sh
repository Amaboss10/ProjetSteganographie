#!/bin/bash

# Script de compilation et d'exécution (mode normal ou decrypt)
cd "$(dirname "$0")"
mkdir -p build
cmake -B build -S .
cmake --build build

# Vérifie si "decrypt" est passé en argument
if [[ "$1" == "decrypt" ]]; then
    ./build/ProjetSteganographieBase decrypt
else
    ./build/ProjetSteganographieBase
fi
