#!/bin/bash
# Kuuntelija Puhtilla: Slurm array -jobi, 8 GPU:ta rinnakkain.
# Kukin task käsittelee joka kahdeksannen tiedoston (--siivu N/8), joten
# tasks eivät koske samoihin biiseihin. Käyttö: sbatch csc/kuuntelija-array.sh [kansio]
# Isommalle kansiolle kasvata --array-riviä ja siivujen määrää yhdessä.
#SBATCH --job-name=kuuntelija-arr
#SBATCH --account=project_XXXXXXX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=1-8
#SBATCH --output=kuuntelija_%A_%a.out

set -euo pipefail

TYOTILA=/scratch/${SLURM_JOB_ACCOUNT}/kuuntelija
KANSIO=${1:-$TYOTILA/musiikki}
SIIVUT=8   # pidä samana kuin --array-rivin yläraja

module purge
module load pytorch ffmpeg

export PATH="$TYOTILA/llama.cpp/build/bin:$PATH"
export HF_HOME=$TYOTILA/hf-cache
# V100:ssa (32 Gt) koko malli mahtuu näytönohjaimeen
export KUUNTELIJA_CPU_MOE=0

cd $TYOTILA
.venv/bin/python kuuntelija30b.py "$KANSIO" --siivu ${SLURM_ARRAY_TASK_ID}/${SIIVUT}
