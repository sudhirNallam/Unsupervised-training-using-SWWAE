#!/bin/bash
#
#SBATCH --job-name=e50p02swwae1lr0001
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=3GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu
#SBATCH --mail-user=psn240@nyu.edu

module purge
module load pytorch/intel/20170125
#module load cuda/8.0.44
module load torchvision/0.1.7

python mnist_model.py --no-cuda --epochs $1 --saveLocation '/results/model'
python mnist_results.py --no-cuda --savedLocation '/results/model_1.t7' --resultsLocation '/results/sample_submission.csv'
