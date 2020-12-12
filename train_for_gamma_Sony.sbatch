#!/bin/bash -e
#
#SBATCH --verbose
#SBATCH --job-name=gamma_Sony
######SBATCH --nodes=1 ####It can run without this line
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:1 #####Without this line, raise ImportError(msg). Failed to load the native TensorFlow runtime.
######If GPU count is 4, with cpus-per-task line commented out, nodes line commented out. #sbatch: error: GPU number 4 is bigger than CPU number 1
######sbatch: error: Batch job submission failed: Unspecified error
######SBATCH --cpus-per-task=4	#CPU cores per thread. For GPU jobs asking for multiple CPUs in a single node? https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm
######SBATCH --ntasks-per-node=4 #Do I need this line for multiple GPUs? Looks like Multicore feature
#SBATCH --mem=5460MB

#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends
#SBATCH --mail-type=fail         # send email if job fails

#module purge all
#module load python/intel/2.7.12
#module load tensorflow/python2.7/1.1.0
#module load scipy/intel/0.19.1
#module load numpy/python2.7/intel/1.14.0
#module load pillow/intel/4.0.0


source ~/.bashrc # source your bashrc to get access to conda.
cd /scratch/ir967/SID/Learning-to-See-in-the-Dark/
conda activate sid2 #commented out module load above since conda does everything for me
python train_for_gamma_Sony.py