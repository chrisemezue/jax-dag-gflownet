#!/bin/bash

for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dibs" "gadget" "mc3" "dag-gfn"
#for baseline in "bcdnets"
do
    #sbatch evaluate_kde.sh $baseline
    sbatch evaluate_kde_100.sh $baseline
    #sbatch evaluate_kde_sachs.sh $baseline

done    
