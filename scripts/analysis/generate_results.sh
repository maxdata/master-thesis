#!/bin/bash

mkdir -p outputs

for experiment in context_sizes failures full_results representations webke zero_shot ; do
    echo $experiment
    python3 "analyse_${experiment}.py" > "outputs/${experiment}.txt"
done
