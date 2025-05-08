#!/bin/bash

# Script de compilation et d'exécution (supporte tous les arguments)
cd "$(dirname "$0")"
mkdir -p build
cmake -B build -S .
cmake --build build

# Transmet tous les arguments à l'exécutable
./build/ProjetSteganographieBase "$@"
