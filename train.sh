#!/bin/bash

for c in 1
do
    for L in 81
    do
        for l_l in 0.1
        do
            for l_z in 0.001
            do
                for l_r in 1.0
                do
                    for l_a in 1.0
                    do
                        python train.py -c config.json --L $L --l_l $l_l --l_z $l_z --l_r $l_r --l_a $l_a --c $c --n "AlphaScale_O2-81-lat$L-L$l_l-Z$l_z-R$l_r-A$l_a-_MNIST-C$c"
                    done
                done
            done
        done
    done
done