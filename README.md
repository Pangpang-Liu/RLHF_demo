# RLHF Demo with TRL

### Minimal End-to-End Reinforcement Learning from Human Feedback Pipeline

This repository provides a minimal end-to-end RLHF pipeline using the Hugging Face `trl` library.

The implementation is intentionally small and fast so the full RLHF workflow can run locally for experimentation and educational purposes.

This demo can be trained using either CPU or GPU by setting `use_cpu=False` for GPU and `use_cpu=True` for CPU.

## Dataset: PRISM Alignment

This demo uses only the dataset [`HannahRoseKirk/prism-alignment`](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment).

PRISM is a human-preference alignment dataset built from conversation trajectories with preferred and non-preferred assistant responses. In this repo:

- SFT uses prompts paired with chosen responses
- reward modeling uses prompt, chosen, and rejected responses
- PPO uses prompts derived from the same PRISM conversations
- DPO uses prompt, chosen, and rejected preference pairs from the same dataset

Both demos train on only tiny subsets of the dataset so the pipeline stays lightweight for a demo.

## RLHF Flow Chart

![RLHF flow chart](./rlhf.png)

## Run RLHF Training Pipeline

```bash
python ppo_demo.py
```

## Run DPO

```bash
python dpo_demo.py
```

Adjust the sample-count constants in [dpo_demo.py](/D:/RLHF_demo-main/dpo_demo.py) and [ppo_demo.py](/D:/RLHF_demo-main/ppo_demo.py) if you want an even smaller or slightly larger demo run.

## Reference

```bibtex
@unpublished{rlhf_revieww,
  title  = {Reinforcement Learning from Human Feedback: A Statistical Perspective},
  author = {Liu, Pangpang and Shi, Chengchun and Sun, Will Wei Sun}
}
```
