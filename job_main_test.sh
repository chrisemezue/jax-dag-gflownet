#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --output=/network/scratch/c/chris.emezue/slurm_sc_100/main_slurmoutput_ate-%j.txt
#SBATCH --error=/network/scratch/c/chris.emezue/slurm_sc_100/main_slurmerror_ate-%j.txt


# Env activate
source /home/mila/c/chris.emezue/gsl-env/bin/activate
BASELINE_FOLDER=/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss100
ATE_DATAFRAME_FOLDER=/home/mila/c/chris.emezue/scratch/ate_estimates_main_100

# Manipulate environment for the needs of this script
unset  SLURM_MEM_PER_GPU    # Clear out the effects of --mem-per-gpu=
unset  SLURM_MEM_PER_NODE   # Clear out the effects of --mem=

ATE_FILES=$(ls -1 ${ATE_DATAFRAME_FOLDER})

# Generate and execute commands
for treatment in T S R Q P O N M L K J I H G F E D C B A; do
#for treatment in B; do
    for outcome in A B C D E F G H I J K L M N O P Q R S T; do
    #for outcome in A; do
        for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dibs" "gadget" "mc3" "dag-gfn"; do
        #for baseline in "bootstrap_ges"; do
        #for baseline in "dag-gfn"; do
            for i in 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 0 1 2 3 4 5 6 7 8 9; do
                if [[ "$treatment" != "$outcome" ]]
                then
                    var=$(echo "$ATE_FILES" | grep -P "${baseline}_.*[^0-9]${i}[,\]].*_${treatment}_${outcome}.*.csv")
                    if [[ "$var" == "" ]]
                    then 
                        FILE=${BASELINE_FOLDER}/${baseline}/${i}/graph.pkl
                        if test -f "$FILE"; then
                            echo $baseline $i $treatment $outcome $BASELINE_FOLDER $ATE_DATAFRAME_FOLDER
                        #else
                        #    echo not exists $baseline $i $treatment $outcome
                        fi
                    fi
                fi
            done
        done
    done
done | parallel -j$SLURM_NTASKS \
srun --immediate --exact --ntasks=1 --nodes=1 \
     --output=/network/scratch/c/chris.emezue/slurm_sc_100/slurmoutput_ate_%j_%s.txt \
     --error=/network/scratch/c/chris.emezue/slurm_sc_100/slurmerror_ate_%j_%s.txt \
     python causal_inference_main.py

