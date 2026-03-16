# RLHF Demo with TRL  
### Minimal End-to-End Reinforcement Learning from Human Feedback Pipeline


This repository provides a **minimal end-to-end RLHF pipeline** using the Hugging Face **TRL** library.


The implementation is intentionally **small and fast** so the full RLHF workflow can run locally for experimentation and educational purposes.

This demo can be trained using either CPU or GPU by setting `use_cpu=False` for GPU and `use_cpu=True` for CPU.

---

# Run RLHF Training Pipeline

python ppo_demo.py

# Run DPO

python dpo_demo.py

