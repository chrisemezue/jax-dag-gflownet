#!/bin/bash

for treatment in 'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T'
#for treatment in 'R' 'T'
do
    for outcome in 'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T'
    #for outcome in 'P' 'E'

    do

        #for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3" "dag-gfn"
        for baseline in "bootstrap_ges"
        #for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dibs" "gadget" "mc3"

        do    
            sbatch job_ci_main.sh $baseline $treatment $outcome

        done
    done
done
