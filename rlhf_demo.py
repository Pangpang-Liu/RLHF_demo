"""
RLHF Demo Script using Hugging Face TRL

This script demonstrates a minimal end-to-end Reinforcement Learning from Human Feedback (RLHF)
pipeline consisting of three stages:

1. Supervised Fine-Tuning (SFT)

2. Reward Model Training

3. PPO Reinforcement Learning

Notes:
- This demo uses small dataset slices and a 0.5B parameter model for fast experimentation.
- Can be trained using either CPU or GPU by setting `use_cpu=False` for GPU.
- Output models are saved in "sft-model/", "reward-model/", and "ppo-model/" directories.
- Designed for educational purposes and quick testing of RLHF workflows.
"""

from trl import SFTTrainer, SFTConfig, RewardTrainer, RewardConfig
from trl.experimental.ppo import PPOTrainer,PPOConfig,AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification,AutoModelForCausalLM
from datasets import load_dataset
from accelerate import PartialState
import time
start=time.time()
use_cpu=False
model_name="Qwen/Qwen2.5-0.5B"
#--- 1. SFT PHASE ---
dataset_sft = load_dataset("trl-lib/Capybara", split=f"train[:10]")
dataset_sft_eval=load_dataset("trl-lib/Capybara", split=f"test[:10]")
sft_trainer = SFTTrainer(
    model=model_name,
    train_dataset=dataset_sft,
    eval_dataset=dataset_sft_eval,
    args=SFTConfig(use_cpu=use_cpu, output_dir="sft-model")
)
sft_trainer.train()
sft_trainer.save_model("sft-model")
# Print Evaluation Metrics
print("SFT Evaluation Results:", sft_trainer.evaluate())

# --- 2. REWARD PHASE ---
dataset_rew = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:10]")
dataset_rew_test=load_dataset("trl-lib/ultrafeedback_binarized", split="test[:10]")
reward_trainer = RewardTrainer(
    model=model_name,
    train_dataset=dataset_rew,
    eval_dataset=dataset_rew_test,
    args=RewardConfig(use_cpu=use_cpu, output_dir="reward-model")
)
reward_trainer.train()
reward_trainer.save_model("reward-model")
# Print Evaluation Metrics
print("Reward Model Evaluation Results:", reward_trainer.evaluate())

# --- 3. PPO PHASE ---
config = PPOConfig(
    use_cpu=use_cpu,
    output_dir="ppo-model"
)
# Load the SFT model we just trained as the policy
model = AutoModelForCausalLM.from_pretrained("sft-model")
tokenizer = AutoTokenizer.from_pretrained("sft-model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
# Reference model for KL penalty
ref_model = AutoModelForCausalLM.from_pretrained("sft-model")
# Load your saved reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("reward-model",num_labels=1)
value_model = AutoModelForSequenceClassification.from_pretrained("reward-model",num_labels=1)
dataset_rl = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness[:20]")
dataset_rl_eval=load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness[20:30]")

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element["prompt"],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=None,
    )

# Compute that only on the main process for faster data processing.
# see: https://github.com/huggingface/trl/pull/1255
with PartialState().local_main_process_first():
    dataset_rl = prepare_dataset(dataset_rl, tokenizer)
    dataset_rl_eval = prepare_dataset(dataset_rl_eval, tokenizer)
# Initialize PPOTrainer
ppo_trainer = PPOTrainer(
    args=config,
    model=model,
    ref_model=ref_model,
    processing_class=tokenizer,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset_rl,
    eval_dataset=dataset_rl_eval
)
ppo_trainer.train()
ppo_trainer.save_model("ppo-model")
end=time.time()
print('total time:',end-start)
