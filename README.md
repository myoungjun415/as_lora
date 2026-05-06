# Adaptive Selection of LoRA Components in Privacy-Preserving Federated Learning

This repository contains the codebase for **Adaptive Selection of LoRA Components in Privacy-Preserving Federated Learning**.

## Environment Setup

```bash
conda create -n as_lora python=3.10 -y
conda activate as_lora
pip install -r requirements.txt
```

## Data Download and Preprocessing

To preprocess all GLUE-style language tasks used in the paper, run:

```bash
python data/prepare_glue.py
python data/prepare_squad.py
python data/prepare_cifar100.py
python data/prepare_tinyimagenet.py
```

## Supported Methods
The repository currently supports the following privacy-preserving federated LoRA methods:
- fedlora
- ffa_lora
- rolora
- as_lora

## Training

### GLUE Tasks

Use `run_glue.py` to train GLUE-style language understanding tasks.

```bash
python run_glue.py --task sst2 --method as_lora --batch_size 128 --epsilon 3 --lr_a 0.5 --lr_b 0.5 --seed 42
```

### SQuAD Tasks

Use `run_squad.py` to train extractive question answering tasks.

```bash
python run_squad.py --task squad11 --method as_lora --batch_size 128 --epsilon 3 --lr_a 0.5 --lr_b 0.5 --seed 42
```

For SQuAD v2.0:
```bash
python run_squad.py --task squad20 --method as_lora --batch_size 128 --epsilon 3 --lr_a 0.5 --lr_b 0.5 --seed 42
```

### Vision Tasks

Use `run_vision.py` to  train vision benchmarks.

```bash
python run_vision.py --task cifar100 --method as_lora --batch_size 128 --epsilon 3 --lr_a 0.5 --lr_b 0.5 --seed 42
```

For Tiny-ImageNet:
```bash
python run_vision.py --task tinyimagenet --method as_lora --batch_size 128 --epsilon 3 --lr_a 0.5 --lr_b 0.5 --seed 42
```
