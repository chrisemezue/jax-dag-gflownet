#!/bin/bash
#SBATCH --array=5-30:5
#SBATCH --job-name=cause-estimation
#SBATCH --mem=50GB
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=unkillable
#SBATCH --error=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/slurmerror_ate_%A_%a.txt
#SBATCH --output=/home/mila/c/chris.emezue/jax-dag-gflownet/slurm_sc/slurmoutput_ate_%A_%a.out.txt


###########cluster information above this line
source /home/mila/c/chris.emezue/gsl-env/bin/activate

#python causal_inference_special_cases.py dibs 25 treatment_effect_with_mediator
python causal_inference_special_cases.py $1 $SLURM_ARRAY_TASK_ID $2
