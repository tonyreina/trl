# TRL Medical Reasoning

Fine-tuning language models for medical reasoning using
[TRL (Transformers Reinforcement Learning)](https://huggingface.co/docs/trl/index)
and [LoRA (Low-Rank Adaptation)](https://tonyreina.github.io/lora/getting-started/what-is-lora/).

## Overview

This project demonstrates efficient fine-tuning of small language models
on medical reasoning tasks using:

- **TRL** - Transformers Reinforcement Learning library for supervised
  fine-tuning
- **LoRA** - Low-Rank Adaptation for memory-efficient training
- [**SmolLM2-135M-Instruct**](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)
  \- Lightweight base model

> [!IMPORTANT]
> **Chain of Thought (CoT):** SmolLM2-135M-Instruct **does not** have chain
> of thought (CoT) reasoning, but I chose it because it was a very small
> model that could be fine-tuned on this dataset in a few hours.

## Project Structure

- `trl_medical_reasoning_training.ipynb` - Fine-tuning notebook with TRL and LoRA
- `trl_medical_reasoning_inference.ipynb` - Model inference and evaluation
- `results/` - Training checkpoints and model artifacts
- `lora_adapter/` - Saved LoRA adapters

> [!TIP]
> **Training UI:** The training logs are sent to [MLFlow](https://mlflow.org).
> During training you can start the MLFlow server locally by running
> `pixi run -e cuda mlflow ui`. The UI will be at `https://localhost:5000`

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Dependencies managed via `pixi.toml`

### Installation

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies
pixi install
```

### Training

Open and run `trl_medical_reasoning_training.ipynb` to:

1. Load and preprocess medical reasoning datasets
2. Configure LoRA adapters and quantization
3. Fine-tune the model using TRL's SFTTrainer
4. Save checkpoints and adapters

### Inference

Use `trl_medical_reasoning_inference.ipynb` to:

- Load trained adapters
- Run inference on medical reasoning queries
- Evaluate model performance

## Documentation

View the full documentation at the
[Jupyter Book site](https://tonyreina.github.io/trl).

## License

Apache v2.0. See [LICENSE](LICENSE) for details.
