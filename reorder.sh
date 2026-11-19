#!/usr/bin/bash
#SBATCH --job-name=reorder
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%x-%j.out

# Reorders Qwen3-1.7B by importance (reorder_qwen3.py), writes
# hf_export/qwen3_1p7b_reordered and compares the untrained level losses of the
# original and reordered 10-level FlexLLaMA.
#   sbatch reorder.sh
#   sbatch reorder.sh --heads    # also reorders the attention heads

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env

cd "$SLURM_SUBMIT_DIR"
python3 reorder_qwen3.py "$@"
