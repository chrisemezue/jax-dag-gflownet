#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

#module load cuda/11.1/cudnn/8.0
#module load singularity/3.7.1
#source /home/mila/c/chris.emezue/gsl-env/bin/activate
module load python/3.8
#module load gcc/8.4.0
source /home/mila/c/chris.emezue/scratch/py38env/bin/activate
export CUDA_VISIBLE_DEVICES=0

#export SLURM_TMPDIR=/home/mila/c/chris.emezue/scratch/SINGULARITY_CDT_TMP_DIR
#echo $SLURM_TMPDIR

#cp /network/scratch/m/mansi.rankawat/gflownet_correct3.simg $SLURM_TMPDIR

python3 get_all_orientations_dag.py

