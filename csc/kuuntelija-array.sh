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
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=1-8
#SBATCH --output=kuuntelija_%A_%a.out

set -euo pipefail

TYOTILA=/scratch/${SLURM_JOB_ACCOUNT}/kuuntelija
KANSIO=${1:-$TYOTILA/musiikki}
SIIVUT=8   # pidä samana kuin --array-rivin yläraja

module purge
module load pytorch ffmpeg

# llama.cpp:n binääri on käännetty CUDA 11:tä vasten (libcudart.so.11.0,
# libcublas.so.11), mutta pytorch tuo cuda/12:n eikä Lmod salli kahta
# cuda-versiota yhtä aikaa. Lisätään siksi CUDA 11.7:n kirjastot suoraan
# LD_LIBRARY_PATHiin (eri sonimet kuin cuda/12:lla, ei törmää; torch pyörii
# tässä CPU:lla). Polun saa: module load gcc/11.3.0 cuda/11.7.0 && echo $CUDA_INSTALL_ROOT
export LD_LIBRARY_PATH="/appl/spack/v018/install-tree/gcc-11.3.0/cuda-11.7.0-zucvj4/lib64:${LD_LIBRARY_PATH:-}"

# CSC:n pytorch-moduuli on Apptainer-kontti, joka nollaa LD_LIBRARY_PATH:n
# sisällään. Kontti-python käynnistää llama-mtmd-cli:n kontin sisällä, joten
# CUDA 11.7:n polku pitää välittää konttiin APPTAINERENV_-etuliitteellä
# (muuten libcudart.so.11.0 ei löydy → exit 127).
export APPTAINERENV_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export SINGULARITYENV_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

export PATH="$TYOTILA/llama.cpp/build/bin:$PATH"
export HF_HOME=$TYOTILA/hf-cache
# V100:ssa (32 Gt) koko malli mahtuu näytönohjaimeen
export KUUNTELIJA_CPU_MOE=0

cd $TYOTILA
.venv/bin/python kuuntelija30b.py "$KANSIO" --siivu ${SLURM_ARRAY_TASK_ID}/${SIIVUT}
