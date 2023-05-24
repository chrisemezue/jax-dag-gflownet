#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc_t_sachs/main_slurmoutput_ate-%j.txt
#SBATCH --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc_t_sachs/main_slurmerror_ate-%j.txt

# Env activate
#source /home/mila/c/chris.emezue/gsl-env/bin/activate
source /home/mila/c/chris.emezue/scratch/py38env/bin/activate


# Generate and execute commands
for treatment in 'Akt' 'Erk' 'Jnk' 'Mek' 'P38' 'PIP2' 'PIP3' 'PKA' 'PKC' 'Plcg' 'Raf'; do
#for treatment in 'Akt'; do
    for outcome in 'Akt' 'Erk' 'Jnk' 'Mek' 'P38' 'PIP2' 'PIP3' 'PKA' 'PKC' 'Plcg' 'Raf'; do
    #for outcome in 'Erk'; do
        for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dibs" "gadget" "mc3" "dag-gfn"; do
        #for baseline in "dag-gfn"; do
            #for i in 5 15 20 25 30; do
            #for i in 5; do
            echo $baseline $treatment $outcome
        done
    done
done | parallel -j$SLURM_NTASKS env -u SLURM_MEM_PER_NODE \
srun --ntasks=1 --nodes=1 --mem=0 \
     --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc_t_sachs/slurmoutput_ate_%j_%s.txt \
     --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc_t_sachs/slurmerror_ate_%j_%s.txt \
     python causal_inference_true_sachs.py


#$SLURM_NTASKS     