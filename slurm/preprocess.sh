#!/bin/bash
#SBATCH -J fineweb_preprocess
#SBATCH -p berzelius-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/preprocess_%j.out
#SBATCH -e logs/preprocess_%j.err

set -e
echo "Job started: $(date)"
echo "Node: $(hostname)"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load buildenv-gcccuda/12.1.1-gcc12.3.0

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd /proj/dl4nlp/users/x_romoj/code/LLM-Labs

echo "Starting preprocessing..."
python lab1_preprocess.py

echo "Job finished: $(date)"
