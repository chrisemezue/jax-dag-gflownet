#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --job-name=kde-evaluate
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --output=/network/scratch/c/chris.emezue/slurm_kde_sachs/main_slurmoutput_ate-%j.txt
#SBATCH --error=/network/scratch/c/chris.emezue/slurm_kde_sachs/main_slurmerror_ate-%j.txt

# Env activate
source /home/mila/c/chris.emezue/scratch/py38env/bin/activate


# Manipulate environment for the needs of this script
unset  SLURM_MEM_PER_GPU    # Clear out the effects of --mem-per-gpu=
unset  SLURM_MEM_PER_NODE   # Clear out the effects of --mem=


# for sachs
export BASELINE_FOLDER=/home/mila/c/chris.emezue/gflownet_sl/tmp/sachs_obs
export ATE_FOLDER=/home/mila/c/chris.emezue/scratch/ate_estimates_sachs
export SCRATCH_FOLDER=/home/mila/c/chris.emezue/scratch/causal_inference/kde_sachs
export NUMBER_OF_NODES=1
export NUMBER_OF_SAMPLES=853

#####


mkdir -p $SCRATCH_FOLDER


# Generate and execute commands
#for treatment in A B C D E F G H I J K L M N O P Q R S T; do
for treatment in 'Akt' 'Erk' 'Jnk' 'Mek' 'P38' 'PIP2' 'PIP3' 'PKA' 'PKC' 'Plcg' 'Raf'; do
#for treatment in 'Akt'; do
    #for outcome in A B C D E F G H I J K L M N O P Q R S T; do
    for outcome in 'Akt' 'Erk' 'Jnk' 'Mek' 'P38' 'PIP2' 'PIP3' 'PKA' 'PKC' 'Plcg' 'Raf'; do
    #for outcome in 'Erk'; do
        for baseline in $1; do
        #for baseline in "dag-gfn"; do
            for i in 5; do
            #for i in 5; do
            # We are not using `i` here
                echo $baseline $i $treatment $outcome $BASELINE_FOLDER $ATE_FOLDER $SCRATCH_FOLDER $NUMBER_OF_NODES $NUMBER_OF_SAMPLES
            done
        done
    done
done | parallel -j1 \
srun --immediate --exact --ntasks=1 --nodes=1 \
     --output=/network/scratch/c/chris.emezue/slurm_kde_sachs/slurmoutput_ate_%j_%s.txt \
     --error=/network/scratch/c/chris.emezue/slurm_kde_sachs/slurmerror_ate_%j_%s.txt \
     python evaluate_ate.py



#-j$SLURM_NTASKS
