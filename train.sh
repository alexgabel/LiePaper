#!/bin/bash

# For various learning rates, run the python train script
for c in 1
do
    for lr in 0.001
    do
        for lra in 0.01
        do
            for l_l in 0.1
            do
                for l_z in 0.001
                do
                    for l_r in 1.0
                    do
                        for l_a in 0.001 1.0 1000.0 0.00001
                        do
                            python train.py -c config.json --lr $lr --lra $lra --l_l $l_l --l_z $l_z --l_r $l_r --l_a $l_a --c $c --n "BIG-81-$lr-$lra-L$l_l-Z$l_z-R$l_r-A$l_a-_MNIST-C$c"
                        done
                    done
                done
            done
        done
    done
done