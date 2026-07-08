#!/bin/bash
# Kuuntelija Puhtilla: yksi GPU-jobi, joka käsittelee koko kansion.
# Käyttö: sbatch csc/kuuntelija-yksi.sh [kansio]
# Kansio-oletus on $TYOTILA/musiikki. Voi lähettää uudelleen jos aika
# loppuu kesken — valmiit biisit ohitetaan.
#SBATCH --job-name=kuuntelija
#SBATCH --account=project_XXXXXXX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=kuuntelija_%j.out

set -euo pipefail

TYOTILA=/scratch/${SLURM_JOB_ACCOUNT}/kuuntelija
KANSIO=${1:-$TYOTILA/musiikki}

module purge
module load pytorch

export PATH="$TYOTILA/llama.cpp/build/bin:$TYOTILA/bin:$PATH"
export HF_HOME=$TYOTILA/hf-cache
# V100:ssa (32 Gt) koko malli mahtuu näytönohjaimeen
export KUUNTELIJA_CPU_MOE=0

cd $TYOTILA
.venv/bin/python kuuntelija30b.py "$KANSIO"
