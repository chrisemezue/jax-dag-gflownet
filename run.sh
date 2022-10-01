#!/bin/bash

#for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3"
for baseline in "dag_gflownet"
do    

    for i in {0..25};
    do
        sbatch job2.sh $baseline $i
    done

done

