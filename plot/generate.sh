#!/usr/bin/bash

set -e

size=1024
kernel=vie
variants=(seq seq_tile seq_opti omp omp_tile omp_opti ocl ocl_opti)
iter=30
export PLATFORM=1

function command {
    echo "./2Dcomp -k $1 -v $2 -s $3 -i $4 -n"
}

function plot_compare {
    index=0
    for variant in "${variants[@]}"
    do
        echo -n "$index \"$variant\" "
        cmd=$(command $kernel $variant $size $iter)
        $cmd |& grep -o "[0-9]*\.[0-9]*"
        index=$((index+1))
    done
}

plot_compare > plot/plot_compare.dat
cat plot/plot_compare.conf | gnuplot
