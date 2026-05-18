#!/bin/bash
#SBATCH --job-name=camchex
#SBATCH --partition=workq         
#SBATCH --gres=gpu:1             
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                
#SBATCH --time=24:00:00
#SBATCH --output=../output/camchex/slurm/%x-%j.out


# Run from the directory where the script was submitted
cd "$SLURM_SUBMIT_DIR"

mkdir -p ../output/camchex/slurm ../output/camchex/checkpoints ../output/camchex/logs

CONFIG_PATH="${CAMCHEX_CONFIG:-../configs/baseline.yaml}"
LOCAL_CONFIG_PATH="${CAMCHEX_LOCAL_CONFIG:-config.local.yaml}"

CONFIG_ARGS=(--config "$CONFIG_PATH")
if [ -f "$LOCAL_CONFIG_PATH" ]; then
    CONFIG_ARGS+=(--config "$LOCAL_CONFIG_PATH")
fi

python -u main.py fit "${CONFIG_ARGS[@]}"
