#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/main_slurmoutput_ate-%j.txt
#SBATCH --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/main_slurmerror_ate-%j.txt

# Env activate
source /home/mila/c/chris.emezue/gsl-env/bin/activate

# Generate and execute commands
for treatment in A B C D E F G H I J K L M N O P Q R S T; do
#for treatment in B; do
    for outcome in A B C D E F G H I J K L M N O P Q R S T; do
    #for outcome in A; do
        for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3" "dag-gfn"; do
        #for baseline in "dag-gfn"; do
            for i in 5 10 15 20 25 30; do
            #for i in 5; do
                echo $baseline $i $treatment $outcome
            done
        done
    done
done | parallel -j$SLURM_NTASKS env -u SLURM_MEM_PER_NODE \
srun --ntasks=1 \
     --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/slurmoutput_ate_%j_%s.txt \
     --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/slurmerror_ate_%j_%s.txt \
     python causal_inference_main.py











# #SBATCH --ntasks=8
# #SBATCH --cpus-per-task=2
# #SBATCH --mem-per-cpu=8G
# #SBATCH --time=7-00:00:00
# #SBATCH --partition=long
# #SBATCH --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/main_slurmoutput_ate.txt 
# #SBATCH --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/main_slurmerror_ate.txt 

# # Env activate
# #source /home/mila/c/chris.emezue/gsl-env/bin/activate

# # Generate and execute commands
# #for treatment in A B C D E F G H I J K L M N O P Q R S T; do
# #for treatment in B; do
# #    for outcome in A B C D E F G H I J K L M N O P Q R S T; do
#     #for outcome in A; do#
#         for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3" "dag-gfn"; do
#         #for baseline in "dag-gfn"; do
#             for i in 5 10 15 20 25 30; do
#             #for i in 5; do
#                 echo $baseline $i $treatment $outcome
#             done
#         done
#     done
# done | parallel -j$SLURM_NTASKS srun --ntasks=1 --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/slurmoutput_ate_%j_%s.txt --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/slurmerror_ate_%j_%s.txt python causal_inference_main.py
# #$SLURM_NTASKS