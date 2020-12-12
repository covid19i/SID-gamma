#!/bin/sh
#

#SBATCH --verbose
#SBATCH --job-name=hw4-1a
#SBATCH --output=1a_slurm_%j.out
#SBATCH --error=1a_slurm_%j.err
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB
#SBATCH --partition="p100_4"
#SBATCH --gres=gpu:1


#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends

module load gcc/6.3.0
module load cuda/9.2.88
nvcc -arch=sm_60 hw4-1a.cu -o hw4-1a -Xcompiler -fopenmp
./hw4-1a
