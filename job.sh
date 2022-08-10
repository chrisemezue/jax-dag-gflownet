#!/bin/bash
#SBATCH --job-name=gsl_big
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/gflownet_sl/slurmerror_2.txt
#SBATCH --output=/home/mila/c/chris.emezue/gflownet_sl/slurmoutput_2.txt


###########cluster information above this line
source /home/mila/c/chris.emezue/gsl-env/bin/activate
#module load anaconda/3
module load python/3.7
module load cuda/11.1/cudnn/8.0
module load pytorch/1.8.1
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mila/c/chris.emezue/gsl-env/lib"
export WANDB_API_KEY=831cb57f73367e89b34e0e6cfdb9e2d143987fcd
python main.py \
--graph erdos_renyi_lingauss \
--num_variables 20 \
--num_samples 100 \
--num_edges 40 \
--n_step 1 \
--batch_size 256 \
--lr 1e-6