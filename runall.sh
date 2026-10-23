#!/usr/bin/bash
mkdir -p ~/jobs
python3 run_experiment.py listcommand $1 | while read config
do
    echo $config
    sbatch $config
done