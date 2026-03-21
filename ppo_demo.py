"""
RLHF Demo Script using Hugging Face TRL

This script demonstrates a minimal end-to-end Reinforcement Learning from Human Feedback (RLHF)
pipeline consisting of three stages:

1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. PPO Reinforcement Learning

Notes:
- This demo uses a tiny subset of HannahRoseKirk/prism-alignment and a 0.5B
  parameter model for fast experimentation.
- This demo can be trained using either CPU or GPU by setting `use_cpu=False`
  for GPU and `use_cpu=True` for CPU.
- Output models are saved in "sft-model/", "reward-model/", and "ppo-model/"
  directories.
- Designed for educational purposes and quick testing of RLHF workflows.
"""

import time

from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from trl import RewardConfig, RewardTrainer, SFTConfig, SFTTrainer
from trl.experimental.ppo import PPOConfig, PPOTrainer

DATASET_NAME = "HannahRoseKirk/prism-alignment"
DATASET_CONFIG = "conversations"

# Keep the demo intentionally small.
SFT_TRAIN_SAMPLES = 32
SFT_EVAL_SAMPLES = 8
REWARD_TRAIN_PAIRS = 32
REWARD_EVAL_PAIRS = 8
PPO_TRAIN_PROMPTS = 32
PPO_EVAL_PROMPTS = 8

RAW_TRAIN_SLICE = "train[:240]"
RAW_EVAL_SLICE = "train[240:320]"


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


def _build_preference_rows(split, max_pairs):
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
                    return rows

            prefix.extend(history[idx:next_idx])
            idx = next_idx

    if not rows:
        raise ValueError(
            "No prompt/chosen/rejected pairs were extracted from PRISM. "
            "Check the dataset config or schema."
        )

    return rows


def build_reward_dataset(split, max_pairs):
    return Dataset.from_list(_build_preference_rows(split, max_pairs))


def build_sft_dataset(split, max_samples):
    preference_rows = _build_preference_rows(split, max_samples)
    return Dataset.from_list(
        [{"text": f'{row["prompt"]} {row["chosen"]}'} for row in preference_rows]
    )


def build_ppo_prompt_dataset(split, max_prompts):
    preference_rows = _build_preference_rows(split, max_prompts)
    return Dataset.from_list([{"prompt": row["prompt"]} for row in preference_rows])


def prepare_dataset(dataset, tokenizer):
    """Pre-tokenize the dataset before training; only collate during training."""

    def tokenize(element):
        outputs = tokenizer(element["prompt"], padding=False)
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=None,
    )


start = time.time()
use_cpu = False
model_name = "Qwen/Qwen2.5-0.5B"

# --- 1. SFT PHASE ---
dataset_sft = build_sft_dataset(RAW_TRAIN_SLICE, SFT_TRAIN_SAMPLES)
dataset_sft_eval = build_sft_dataset(RAW_EVAL_SLICE, SFT_EVAL_SAMPLES)
sft_trainer = SFTTrainer(
    model=model_name,
    train_dataset=dataset_sft,
    eval_dataset=dataset_sft_eval,
    args=SFTConfig(
        use_cpu=use_cpu,
        output_dir="sft-model",
        dataset_text_field="text",
    ),
)
sft_trainer.train()
sft_trainer.save_model("sft-model")
print("SFT Evaluation Results:", sft_trainer.evaluate())

# --- 2. REWARD PHASE ---
dataset_rew = build_reward_dataset(RAW_TRAIN_SLICE, REWARD_TRAIN_PAIRS)
dataset_rew_eval = build_reward_dataset(RAW_EVAL_SLICE, REWARD_EVAL_PAIRS)
reward_trainer = RewardTrainer(
    model=model_name,
    train_dataset=dataset_rew,
    eval_dataset=dataset_rew_eval,
    args=RewardConfig(use_cpu=use_cpu, output_dir="reward-model"),
)
reward_trainer.train()
reward_trainer.save_model("reward-model")
print("Reward Model Evaluation Results:", reward_trainer.evaluate())

# --- 3. PPO PHASE ---
config = PPOConfig(use_cpu=use_cpu, output_dir="ppo-model")
model = AutoModelForCausalLM.from_pretrained("sft-model")
tokenizer = AutoTokenizer.from_pretrained("sft-model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
ref_model = AutoModelForCausalLM.from_pretrained("sft-model")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "reward-model", num_labels=1
)
value_model = AutoModelForSequenceClassification.from_pretrained(
    "reward-model", num_labels=1
)
dataset_rl = build_ppo_prompt_dataset(RAW_TRAIN_SLICE, PPO_TRAIN_PROMPTS)
dataset_rl_eval = build_ppo_prompt_dataset(RAW_EVAL_SLICE, PPO_EVAL_PROMPTS)

# Compute that only on the main process for faster data processing.
with PartialState().local_main_process_first():
    dataset_rl = prepare_dataset(dataset_rl, tokenizer)
    dataset_rl_eval = prepare_dataset(dataset_rl_eval, tokenizer)

ppo_trainer = PPOTrainer(
    args=config,
    model=model,
    ref_model=ref_model,
    processing_class=tokenizer,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset_rl,
    eval_dataset=dataset_rl_eval,
)
ppo_trainer.train()
ppo_trainer.save_model("ppo-model")

end = time.time()
print("total time:", end - start)
