"""
DPO Demo Script using Hugging Face TRL

Notes:
- This demo uses small dataset slices and a 0.5B parameter model for fast experimentation.
- This demo can be trained using either CPU or GPU by setting `use_cpu=False` for GPU and `use_cpu=True` for CPU.
- Output models are saved in "dpo-model/" directory.
- Designed for educational purposes and quick testing of RLHF workflows.
"""

from datasets import load_dataset
from trl import DPOTrainer,DPOConfig
import time

start=time.time()
use_cpu=True
dataset_train = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:10]")
dataset_eval=load_dataset("trl-lib/ultrafeedback_binarized", split="test[:10]")
model_name="Qwen/Qwen2.5-0.5B"
trainer = DPOTrainer(
    model=model_name,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    args=DPOConfig(use_cpu=use_cpu, output_dir="dpo-model")
)
trainer.train()
trainer.save_model("dpo-model")
end=time.time()
print('total time:',end-start)