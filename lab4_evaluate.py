"""
Lab 4: Evaluate the GPT-2 model on SciQ test and ARC-Easy test.

Evaluation method: log-likelihood scoring (no generation needed).
For each multiple-choice question, score each answer choice as:
    score = sum(log P(answer_token | context)) / num_answer_tokens
Pick the highest-scoring choice and compare to gold label.

Usage:
    # Evaluate base (pre-trained only) model:
    python lab4_evaluate.py --checkpoint checkpoints/ckpt_final.pt --name "base"

    # Evaluate fine-tuned model:
    python lab4_evaluate.py --checkpoint checkpoints/finetune/finetune-sciq-pretrained_epoch5.pt --name "finetuned"

    # Evaluate random-init fine-tuned model:
    python lab4_evaluate.py --checkpoint checkpoints/finetune/finetune-sciq-random-init_epoch5.pt --name "random_init"
"""

import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint to evaluate")
parser.add_argument("--name",       type=str, default="model",
                    help="Label for this model in results (e.g. 'base', 'finetuned')")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
enc    = tiktoken.get_encoding("gpt2")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

config = GPTConfig()
model  = GPT(config).to(device)
ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded: {args.checkpoint}")

# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_choice(context: str, choice: str) -> float:
    """
    Normalized log P(choice | context).
    Divides by number of answer tokens to avoid bias toward shorter answers.
    """
    ctx_ids    = enc.encode_ordinary(context)
    choice_ids = enc.encode_ordinary(" " + choice)
    all_ids    = ctx_ids + choice_ids

    x = torch.tensor(all_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
    y = torch.tensor(all_ids[1:],  dtype=torch.long).unsqueeze(0).to(device)

    # Pass y to get full-sequence logits (not just last position)
    logits, _ = model(x, y)
    log_probs  = F.log_softmax(logits, dim=-1)

    # Only score the answer portion
    start     = len(ctx_ids) - 1
    choice_lp = log_probs[0, start:, :]
    choice_y  = y[0, start:]

    score = choice_lp[range(len(choice_y)), choice_y].sum().item()
    return score / len(choice_ids)


# ---------------------------------------------------------------------------
# SciQ evaluation
# ---------------------------------------------------------------------------

def evaluate_sciq(split: str = "test") -> dict:
    """
    Evaluates on SciQ multiple-choice.
    Context = support passage + question.
    4 choices: correct_answer, distractor1, distractor2, distractor3.
    """
    dataset = load_dataset("allenai/sciq", split=split)
    correct = 0

    for item in tqdm(dataset, desc=f"SciQ {split}"):
        support  = item["support"].strip()
        question = item["question"].strip()
        context  = f"{support} {question}" if support else question

        choices = [
            item["correct_answer"].strip(),
            item["distractor1"].strip(),
            item["distractor2"].strip(),
            item["distractor3"].strip(),
        ]

        scores   = [score_choice(context, c) for c in choices]
        pred_idx = scores.index(max(scores))

        if pred_idx == 0:   # index 0 is always the correct answer in SciQ
            correct += 1

    accuracy = correct / len(dataset)
    return {
        "dataset":  f"SciQ ({split})",
        "total":    len(dataset),
        "correct":  correct,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# ARC-Easy evaluation
# ---------------------------------------------------------------------------

def evaluate_arc_easy(split: str = "test") -> dict:
    """
    Evaluates on ARC-Easy multiple-choice.
    Context = question only (no support passage in ARC).
    Choices are stored in item["choices"]["text"], correct label in item["answerKey"].
    Handles both 4-choice and occasionally 3-choice questions.
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    correct = 0

    for item in tqdm(dataset, desc=f"ARC-Easy {split}"):
        question     = item["question"].strip()
        choices_text = item["choices"]["text"]
        choices_label = item["choices"]["label"]
        answer_key   = item["answerKey"]

        # Find correct index
        try:
            correct_idx = choices_label.index(answer_key)
        except ValueError:
            # Some ARC examples use "1","2","3","4" instead of "A","B","C","D"
            label_map   = {"1": "A", "2": "B", "3": "C", "4": "D"}
            answer_key  = label_map.get(answer_key, answer_key)
            correct_idx = choices_label.index(answer_key)

        scores   = [score_choice(question, c.strip()) for c in choices_text]
        pred_idx = scores.index(max(scores))

        if pred_idx == correct_idx:
            correct += 1

    accuracy = correct / len(dataset)
    return {
        "dataset":  f"ARC-Easy ({split})",
        "total":    len(dataset),
        "correct":  correct,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Run both evaluations and print results
# ---------------------------------------------------------------------------

print(f"\n{'='*55}")
print(f"  Model: {args.name}  ({args.checkpoint})")
print(f"{'='*55}")

results = []
results.append(evaluate_sciq("test"))
results.append(evaluate_arc_easy("test"))

print(f"\n{'Dataset':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("-" * 50)
for r in results:
    print(f"{r['dataset']:<20} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>9.1%}")
print(f"{'Random baseline':<20} {'—':>8} {'—':>8} {'25.0%':>10}")
print("-" * 50)

# Save results to file
out_path = f"results_{args.name}.txt"
with open(out_path, "w") as f:
    f.write(f"Model: {args.name}\nCheckpoint: {args.checkpoint}\n\n")
    f.write(f"{'Dataset':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}\n")
    f.write("-" * 50 + "\n")
    for r in results:
        f.write(f"{r['dataset']:<20} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>9.1%}\n")
print(f"\nResults saved to {out_path}")
