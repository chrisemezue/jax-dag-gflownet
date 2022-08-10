#!/bin/bash

for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3"
do    

    sbatch job2.sh $baseline

done

