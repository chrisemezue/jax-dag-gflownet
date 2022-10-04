#!/bin/bash

#for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3"
for baseline in "dag_gflownet"
do    

    bash job2.sh $baseline 

done

