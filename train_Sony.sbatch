#!/bin/bash
#
#SBATCH --job-name=SID_train_Sony
#SBATCH --nodes=1
#SBATCH --time=0:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3000MB

#module purge all
#module load python/intel/2.7.12
#module load tensorflow/python2.7/1.1.0
#module load scipy/intel/0.19.1
#module load numpy/python2.7/intel/1.14.0
#module load pillow/intel/4.0.0


source ~/.bashrc # source your bashrc to get access to conda.
cd /scratch/ir967/SID/Learning-to-See-in-the-Dark/
conda activate sid2
python train_Sony.py