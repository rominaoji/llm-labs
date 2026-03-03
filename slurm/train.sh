#!/bin/bash
#
#SBATCH -J fineweb_train
#SBATCH --gpus 1
#SBATCH -C "thin"
#SBATCH -t 0-48:00:00
#



# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Activate your conda/venv environment
# conda activate llmlab   # uncomment and adjust to your env name

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd /proj/dl4nlp/users/x_romoj/code/LLM-Labs

# WANDB_RUN_ID="lvqwdi49"   # get this from the W&B URL of your first run
# CKPT="checkpoints/ckpt_step008000.pt"

torchrun \
    --standalone \
    --nproc_per_node=1 \
    lab2_train.py 

    #--resume $CKPT --wandb_id $WANDB_RUN_ID

echo "Job finished: $(date)"
