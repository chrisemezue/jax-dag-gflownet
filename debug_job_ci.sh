#!/bin/bash
#SBATCH --array=5-30:5
#SBATCH --job-name=cause-estimation
#SBATCH --mem=50GB
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --partition=unkillable


###########cluster information above this line
source /home/mila/c/chris.emezue/gsl-env/bin/activate

#python -m pdb testgr.py dibs 25
python -m pdb testgr.py dag_gflownet 5
