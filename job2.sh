#!/bin/bash
#SBATCH --job-name=testgr
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=unkillable
#SBATCH --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurmerror_ate_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurmoutput_ate_%j.txt


###########cluster information above this line
source /home/mila/c/chris.emezue/gsl-env/bin/activate

python testgr.py $1