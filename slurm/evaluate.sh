#!/bin/bash
#SBATCH -J sciq_evaluate
#SBATCH -p berzelius
#SBATCH --gpus 1
#SBATCH -C "thin"
#SBATCH -t 0-01:00:00
#SBATCH -o logs/evaluate_%j.out
#SBATCH -e logs/evaluate_%j.err

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

echo "=========================================="
echo " Evaluating base (pre-trained only) model"
echo "=========================================="
python lab4_evaluate.py \
    --checkpoint checkpoints/ckpt_final.pt \
    --name "base_pretrained"

echo ""
echo "=========================================="
echo " Evaluating fine-tuned model (epoch 5)"
echo "=========================================="
python lab4_evaluate.py \
    --checkpoint checkpoints/finetune/finetune-sciq-pretrained_epoch5.pt \
    --name "finetuned_pretrained"


echo "=========================================="
echo " Evaluating random+init model (epoch 5)"
echo "=========================================="
python lab4_evaluate.py \
    --checkpoint checkpoints/finetune/finetune-sciq-random-init_epoch5.pt \
    --name "finetuned_random_init"


echo ""
echo "Job finished: $(date)"
