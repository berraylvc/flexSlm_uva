#!/usr/bin/bash
#SBATCH --job-name=upload
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=./jobs/%x-%j.out

# Copies the project to SURFdrive (rclone remote "surfdrive", folder FlexSLM):
#   1. best-model.ckpt of the runs whose checkpoints are on scratch (deleted after 14 days)
#   2. best-model.ckpt of the other runs
#   3. saved_models: final weights of every run, incl. the runs without a checkpoint
#   4. code, job logs, results and data; hf_export without the weights
# rclone copy skips files that are already there, so the job can simply be resubmitted.
#   sbatch upload.sh

cd "$SLURM_SUBMIT_DIR"
DEST=surfdrive:FlexSLM
OPTS="--transfers 2 --checkers 4 --retries 5 --low-level-retries 20 --stats 10m --stats-one-line -v"

rclone copy "/scratch-shared/$USER/checkpoints" "$DEST/checkpoints" --include best-model.ckpt $OPTS
rclone copy checkpoints "$DEST/checkpoints" --include best-model.ckpt $OPTS
rclone copy saved_models "$DEST/saved_models" $OPTS
rclone copy . "$DEST/repo" --exclude "/checkpoints/**" --exclude "/saved_models/**" --exclude "/hf_export/**" \
    --exclude "/wandb/**" --exclude "/.tmp/**" $OPTS
rclone copy hf_export "$DEST/repo/hf_export" --exclude "*.safetensors" $OPTS
echo "UPLOAD DONE"
rclone size "$DEST"
