#!bin/bash


export folder=/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20

for baseline in bcdnets  bootstrap_ges  bootstrap_pc  dag-gfn  dibs  gadget  mc3
do
    cd $folder/$baseline
    for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    do
        cd $folder/$baseline/$seed
        rm -rf kde/
    done
done
