#!/bin/bash
#SBATCH -J liar_llm
#SBATCH -p cas_v100nv_4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time 50:30:00
#SBATCH --comment pytorch
#SBATCH -o /scratch/x2889a01/trlx/logs/output_%x_%j.out
#SBATCH -e /scratch/x2889a01/trlx/logs/output_%x_%j.err

source /scratch/x2889a01/.bashrc
export CONDA_ENVS_PATH=/scratch/x2889a01/.conda/envs
export CONDA_PKGS_DIRS=/scratch/x2889a01/.conda/pkgs
conda activate trlx

WORKSPACE_PATH=$HOME/trlx

cd $WORKSPACE_PATH

echo "START"

srun python $WORKSPACE_PATH/rlhf_medical.py --gpu-list 0

echo "DONE"

