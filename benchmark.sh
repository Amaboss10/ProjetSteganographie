#!/bin/bash

echo "Type,Time_ms" > performance_results.csv

for i in {1..3}; do
    ./run.sh            | grep "Temps CPU" | awk '{print "ENCODE_CPU," $4}' >> performance_results.csv
    ./run.sh gpu        | grep "Temps CUDA" | awk '{print "ENCODE_GPU," $5}' >> performance_results.csv
    ./run.sh decrypt    | grep "Temps CPU" | awk '{print "DECODE_CPU," $4}' >> performance_results.csv
    ./run.sh decrypt gpu  | grep "Temps CPU" | awk '{print "DECODE_GPU," $4}' >> performance_results.csv
    ./run.sh decrypt cuda | grep "Temps GPU" | awk '{print "DECODE_CUDA," $4}' >> performance_results.csv
done

echo "Tous les tests sont terminés. Résultats enregistrés dans performance_results.csv"
