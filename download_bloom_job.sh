#!/usr/bin/bash
#SBATCH --job-name=download_gneissweb_bloom
#SBATCH --partition=staging
#SBATCH --time=02:00:00
#SBATCH --output=./jobs/%x-%j.out

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my_env
export HF_TOKEN=$(cat "$HOME/.hf_token")

python3 download_gneissweb_bloom.py
