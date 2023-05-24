#!/bin/bash

for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dibs" "gadget" "mc3" "dag-gfn";
#for baseline in "bootstrap_ges"
do
    #bash job_sachs.sh $baseline
    sbatch job_sachs.sh $baseline

done    
