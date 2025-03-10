#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --output=log.%x.%j.out
#SBATCH --job-name=prompt_exp
#SBATCH --time=3:0:0
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
module load python/3.12-conda
conda activate lgar
echo "Loaded Anaconda"
cd "/path/to/project/"
python3 -u ./implementation/src/scripts/run_prompt_experiment.py --index $SLURM_ARRAY_TASK_ID