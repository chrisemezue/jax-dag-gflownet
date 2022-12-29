#!/bin/bash

#for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3"
#for baseline in "dibs" "bootstrap_pc"
for baseline in "dibs"
do    
    sbatch job2.sh $baseline 

done

