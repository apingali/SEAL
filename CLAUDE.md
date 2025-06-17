# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SEAL (Self-Adapting Language Models) is a framework for training language models via RL to generate self-edits (finetuning data and other update directives for themselves) in response to new inputs. The repository contains two main experimental domains:

- `few-shot/`: Adapting to new tasks from few-shot examples (uses ARC-AGI dataset)
- `knowledge-incorporation/`: Incorporating new factual knowledge (uses SQuAD dataset)

## Environment Setup

```bash
# Create and activate virtual environment
conda create -n seal_env python=3.12
conda activate seal_env

# Install dependencies
pip install -r requirements.txt

# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Common Development Commands

### Few-Shot Domain (ARC-AGI)

```bash
# Change to few-shot directory
cd few-shot

# Train base model on 12 problems (RL Iteration 1)
python self-edit.py \
    --experiment_name=training_set_iteration_1 \
    --challenge_file=data/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=data/arc-agi_training_solutions_filtered_1B_training_set.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --n_tasks=12 \
    --n_self_edits_per_task=15

# Evaluate trained LoRAs
python eval-self-edits.py \
    --experiment_folder=${TTI_DIR}/training_set_iteration_1 \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/training_set_iteration_1 \
    --temperature=0 \
    --n_sample=1 \
    --data_file=data/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=data/arc-agi_training_solutions_filtered_1B_training_set.json

# Run RestEM training
python BC-self-edit.py \
    --configs_and_indices=${LORA_DIR}/self-edit/training_set_iteration_1/final_configs_and_indices.json \
    --results=${LORA_DIR}/self-edit/training_set_iteration_1/final_results.json \
    --model_name=meta-llama/Llama-3.2-1B-Instruct

# Evaluate baseline model
python eval-self-edits-baseline.py \
    --experiment_folder=${TTI_DIR}/eval_base_model \
    --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct
```

### Knowledge Incorporation Domain

```bash
# Create synthetic data
sbatch knowledge-incorporation/scripts/make_squad_data.sh

# Start TTT server (requires GPU)
sbatch knowledge-incorporation/scripts/TTT_server.sh

# Query server for RL training or evaluation
sbatch knowledge-incorporation/scripts/query_server.sh

# Build SFT dataset for RL training
python3 knowledge-incorporation/src/EM/build_SFT_dataset.py <path/to/result/of/run.json>

# Train SFT model
sbatch knowledge-incorporation/scripts/train_SFT.sh

# Run continual self-edits experiment
sbatch knowledge-incorporation/scripts/continual_self_edits.sh
```

## Architecture Overview

### Few-Shot Domain
- **ARC Library** (`few-shot/arclib/`): Core classes for ARC tasks (Grid, Example, Task)
- **Inference Engines** (`few-shot/inference/`): Multiple engine implementations (HF, vLLM, LoRA variants)
- **Self-Edit Pipeline**: `self-edit.py` generates self-edits, `BC-self-edit.py` performs RestEM training
- **Evaluation**: `eval-self-edits.py` and `eval-self-edits-baseline.py` for model evaluation

### Knowledge Incorporation Domain
- **Data Generation** (`src/data_generation/`): Creates synthetic SQuAD-based datasets
- **TTT Server** (`src/inner/TTT_server.py`): ZMQ server for training temporary LoRA adapters
- **Query System** (`src/query/`): Client for querying TTT server in single/multi-passage settings
- **RL Training** (`src/EM/`): RestEM implementation for self-edit training
- **Continual Learning** (`src/continual/`): Continual self-edits experiments

### Key Components
- **LoRA Integration**: Uses PEFT library for efficient fine-tuning with configurable ranks
- **vLLM Support**: Multiple inference engines supporting vLLM for fast inference
- **ZMQ Communication**: Knowledge incorporation uses ZeroMQ for client-server architecture
- **SLURM Integration**: All scripts include SLURM directives (update before use)

## Important Notes

- All experiments use meta-llama/Llama-3.2-1B-Instruct as the base model
- Update SLURM directives in .sh files before running on your system
- TTT server requires GPU resources and proper port configuration
- Environment variables (TTI_DIR, LORA_DIR, DATA_DIR) need to be set for some scripts
- The framework supports both single-passage and multi-passage knowledge incorporation