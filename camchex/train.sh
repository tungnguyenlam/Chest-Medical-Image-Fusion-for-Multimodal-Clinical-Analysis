#!/bin/bash
#SBATCH --job-name=camchex
#SBATCH --partition=workq         
#SBATCH --gres=gpu:1             
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out 


# Run from the directory where the script was submitted
cd "$SLURM_SUBMIT_DIR"

python -u main.py fit --config config.yaml