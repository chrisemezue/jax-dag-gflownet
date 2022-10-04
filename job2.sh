#!/bin/bash
#SBATCH --array=5-30:5
#SBATCH --job-name=cause-estimation
#SBATCH --mem=50GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --partition=unkillable
#SBATCH --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm2/slurmerror_ate_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm2/slurmoutput_ate_%j.txt


###########cluster information above this line
source /home/mila/c/chris.emezue/gsl-env/bin/activate

#python testgr.py $1 $2
python testgr.py $1 $SLURM_ARRAY_TASK_ID
