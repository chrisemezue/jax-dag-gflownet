#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc_100/main_slurmoutput_ate-%j.txt
#SBATCH --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc_100/main_slurmerror_ate-%j.txt


# Env activate
source /home/mila/c/chris.emezue/gsl-env/bin/activate


# Manipulate environment for the needs of this script
unset  SLURM_MEM_PER_GPU    # Clear out the effects of --mem-per-gpu=
unset  SLURM_MEM_PER_NODE   # Clear out the effects of --mem=
export SLURM_STDERRMODE="$HOME/jax-dag-gflownet/slurm_sc_100/slurmerror_ate_%j_%s.txt"
export SLURM_STDOUTMODE="$HOME/jax-dag-gflownet/slurm_sc_100/slurmoutput_ate_%j_%s.txt"


# Generate and execute commands
for treatment in A B C D E F G H I J K L M N O P Q R S T; do
#for treatment in B; do
    for outcome in A B C D E F G H I J K L M N O P Q R S T; do
    #for outcome in A; do
        for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dibs" "gadget" "mc3" "dag-gfn"; do
        #for baseline in "bootstrap_ges"; do
        #for baseline in "dag-gfn"; do
            for i in 5 10 15 20 25 30; do
            #for i in 5 15 20 25 30; do
            #for i in 5; do
                echo $baseline $i $treatment $outcome
            done
        done
    done
done | parallel -j$SLURM_NTASKS \
srun --immediate --exact --ntasks=1 --nodes=1  \
     python causal_inference_main_100.py
