#!/bin/bash
#SBATCH -J GeonHee
#SBATCH -p cas_v100nv_4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time 40:30:00
#SBATCH --comment pytorch
#SBATCH -o /scratch/x2889a02/GoodLiar/logs/output_%x_%j.out
#SBATCH -e /scratch/x2889a02/GoodLiar/logs/output_%x_%j.err

source /scratch/x2889a02/.bashrc
export CONDA_ENVS_PATH=/scratch/x2889a02/.conda/envs
export CONDA_PKGS_DIRS=/scratch/x2889a02/.conda/pkgs
conda activate demo

WORKSPACE_PATH=$HOME/GoodLiar

cd $WORKSPACE_PATH

echo "START"

srun python $WORKSPACE_PATH/main_on_the_fly.py --gpu-list 0

echo "DONE"

