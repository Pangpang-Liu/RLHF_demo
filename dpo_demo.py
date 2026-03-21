"""
DPO Demo Script using Hugging Face TRL

Notes:
- This demo uses a tiny subset of HannahRoseKirk/prism-alignment and a 0.5B
  parameter model for fast experimentation.
- This demo can be trained using either CPU or GPU by setting `use_cpu=False` for GPU and `use_cpu=True` for CPU.
- Output models are saved in "dpo-model/" directory.
- Designed for educational purposes and quick testing of RLHF workflows.
"""

from datasets import Dataset, load_dataset
from trl import DPOConfig, DPOTrainer
import time

DATASET_NAME = "HannahRoseKirk/prism-alignment"
DATASET_CONFIG = "conversations"

# Keep the demo intentionally small.
TRAIN_PAIRS = 32
EVAL_PAIRS = 8
RAW_TRAIN_SLICE = "train[:200]"
RAW_EVAL_SLICE = "train[200:260]"


def _conversation_history(example):
    for key in ("conversation_history", "conversation", "messages"):
        if key in example and example[key]:
            return example[key]
    return []


def _render_prompt(history):
    prompt_lines = []
    for turn in history:
        role = str(turn.get("role", "user")).strip().title()
        content = str(turn.get("content", "")).strip()
        if content:
            prompt_lines.append(f"{role}: {content}")
    prompt_lines.append("Assistant:")
    return "\n".join(prompt_lines)


def build_prism_preferences(split, max_pairs):
    raw_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    rows = []

    for example in raw_dataset:
        history = _conversation_history(example)
        prefix = []
        idx = 0

        while idx < len(history):
            turn = history[idx]
            role = str(turn.get("role", "")).lower()

            if role != "user":
                prefix.append(turn)
                idx += 1
                continue

            branch_candidates = []
            next_idx = idx + 1
            while next_idx < len(history):
                next_turn = history[next_idx]
                if str(next_turn.get("role", "")).lower() != "model":
                    break
                content = str(next_turn.get("content", "")).strip()
                chosen_flag = next_turn.get("if_chosen")
                if content and chosen_flag is not None:
                    branch_candidates.append(
                        {"content": content, "if_chosen": bool(chosen_flag)}
                    )
                next_idx += 1

            chosen = next(
                (item["content"] for item in branch_candidates if item["if_chosen"]),
                None,
            )
            rejected = next(
                (item["content"] for item in branch_candidates if not item["if_chosen"]),
                None,
            )

            if chosen and rejected:
                rows.append(
                    {
                        "prompt": _render_prompt(prefix + [turn]),
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )
                if len(rows) >= max_pairs:
                    return Dataset.from_list(rows)

            prefix.extend(history[idx:next_idx])
            idx = next_idx

    if not rows:
        raise ValueError(
            "No prompt/chosen/rejected pairs were extracted from PRISM. "
            "Check the dataset config or schema."
        )

    return Dataset.from_list(rows)


start = time.time()
use_cpu = True
dataset_train = build_prism_preferences(RAW_TRAIN_SLICE, TRAIN_PAIRS)
dataset_eval = build_prism_preferences(RAW_EVAL_SLICE, EVAL_PAIRS)
model_name = "Qwen/Qwen2.5-0.5B"
trainer = DPOTrainer(
    model=model_name,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    args=DPOConfig(use_cpu=use_cpu, output_dir="dpo-model")
)
trainer.train()
trainer.save_model("dpo-model")
end = time.time()
print("total time:", end - start)
