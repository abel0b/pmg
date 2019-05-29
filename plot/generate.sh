#!/usr/bin/bash

set -e

size=1024
kernel=vie
variants=(seq seq_tile seq_opti omp omp_tile omp_opti ocl)
omp_variants=(omp omp_tile omp_task_tile omp_task_opti)
iter=30
export PLATFORM=1

function command {
    echo "OMP_NUM_THREADS=$5 ./2Dcomp -k $1 -v $2 -s $3 -i $4 -n -a $6 -g 16"
}

function plot_compare {
    index=0
    for variant in "${variants[@]}"
    do
        echo -n "$index \"$variant\" "
        cmd=$(command $kernel $variant $size $iter 24 $1)
        eval $cmd |& grep -o "[0-9]*\.[0-9]*"
        index=$((index+1))
    done
}

function plot_compare_conf {
    echo "set terminal png size 1200,800"
    echo "set output \"plot/plot_compare$1.png\""
    echo "set xlabel \"variante\""
    echo "set ylabel \"durée (microsecondes)\""
    echo "set boxwidth 0.5"
    echo "set style fill solid"
    echo "plot \"plot/plot_compare.dat\" using 1:3:xtic(2) with boxes"
}

function speed_up_openmp {
    for nbthreads in $(seq 1 100)
    do
        echo -n "$nbthreads"
        for variant in "${omp_variants[@]}"
        do
            echo -n " "
            cmd=$(command $kernel $variant $1 $iter $nbthreads $2)
            eval $cmd #|& grep -o "[0-9]*\.[0-9]*" | tr -d "\n"
        done
        echo
    done
}

function speed_up_openmp_conf {
    echo "set terminal png size 1200,800"
    echo "set output \"plot/speed_up_openmp$1$2.png\""
    index=0
    datafile="plot/speed_up_openmp.dat"
    plt="plot"
    for variant in "${omp_variants[@]}"
    do
        if [[ ! $index == "0" ]]
        then
            plt="$plt,"
        fi
        plt="$plt "
        plt="$plt\"$datafile\" using 1:$((index+2)) with lines title \"$variant\""
        index=$((index+1))
    done
    echo "$plt"
}

function speed_up_opencl {
    for nbthreads in $(seq 1 20)
    do
        echo -n "$nbthreads "
        cmd=$(command $kernel ocl $1 $iter $nbthreads $2)
        eval $cmd |& grep -o "[0-9]*\.[0-9]*"
    done
}

function speed_up_opencl_conf {
    echo "set terminal png size 1200,800"
    echo "set output \"plot/speed_up_opencl$1$2.png\""
    echo "plot \"plot/speed_up_opencl.dat\" using 1:2 with lines"
}

#plot_compare guns > plot/plot_compare.dat
#plot_compare_conf guns > plot/plot_compare.conf
#cat plot/plot_compare.conf | gnuplot

#plot_compare random > plot/plot_compare.dat
#plot_compare_conf random > plot/plot_compare.conf
#cat plot/plot_compare.conf | gnuplot

#speed_up_openmp 512 guns > plot/speed_up_openmp.dat
#speed_up_openmp_conf 512 guns > plot/speed_up_openmp.conf
#cat plot/speed_up_openmp.conf | gnuplot

#speed_up_openmp 512 random > plot/speed_up_openmp.dat
#speed_up_openmp_conf 512 random > plot/speed_up_openmp.conf
#cat plot/speed_up_openmp.conf | gnuplot

#speed_up_openmp 4096 guns > plot/speed_up_openmp.dat
#speed_up_openmp_conf 4096 guns > plot/speed_up_openmp.conf
#cat plot/speed_up_openmp.conf | gnuplot

speed_up_opencl 512 guns > plot/speed_up_opencl.dat
speed_up_opencl_conf 512 guns > plot/speed_up_opencl.conf
cat plot/speed_up_opencl.conf | gnuplot
