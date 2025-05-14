#!/bin/bash

# Crée et active l'environnement
python3 -m venv venv
source venv/bin/activate

# Installe les dépendances compatibles
pip install numpy pandas matplotlib

# Lance le script de graphique
python3 plot_performance.py
