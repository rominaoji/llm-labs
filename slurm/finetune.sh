#!/bin/bash
#SBATCH -J sciq_finetune
#SBATCH -p berzelius
#SBATCH --gpus 1
#SBATCH -C "thin"
#SBATCH -t 0-06:00:00
#SBATCH -o logs/finetune_%j.out
#SBATCH -e logs/finetune_%j.err

set -e
echo "Job started: $(date)"
echo "Node: $(hostname)"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# conda activate llmlab

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd /proj/dl4nlp/users/x_romoj/code/LLM-Labs

CKPT="checkpoints/ckpt_final.pt"   # <-- set to your final pre-trained checkpoint

# Experiment 1: pre-trained init (main result)
# python lab3_finetune.py --checkpoint $CKPT

# Experiment 2: random init baseline (run separately or comment out above)
python lab3_finetune.py --checkpoint $CKPT --random_init

echo "Job finished: $(date)"
