# BCI_code_Implementation
````markdown
# BCI Code Implementation

This repository contains the implementation of DANet for motor-imagery EEG domain adaptation using the MOABB BNCI2014‐001 dataset.

---

## 1. Environment Setup

Create and activate a new Conda environment with Python 3.10:

```bash
conda create -n bci_env python=3.10 -y
conda activate bci_env
````

---

## 2. Core Dependencies

Install the core libraries:

```bash
# NumPy
conda install numpy
```

* **CUDA 11.8**

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

* **CUDA 12.6**

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

* **CUDA 12.8**

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```
```bash
# scikit-learn (for metrics, e.g. Cohen’s kappa)
conda install conda-forge::scikit-learn
# tqdm (progress bars)
conda install conda-forge::tqdm
# MOABB
pip install moabb
# Weights & Biases (optional experiment logging ,but have to install)
conda install conda-forge::wandb
```


## 3. Running the Training Script
Once all dependencies are installed, run:

```bash
python moabb_train.py --wandb False
```

* `--wandb False` disables Weights & Biases logging.
* Other arguments (epochs, batch size, learning rates, etc.) can be passed as flags. See `python moabb_train.py --help` for details.

---

## 4. Directory Structure

```
.
├── moabb_train.py      # Training script
├── model.py            # DANet, FeatureExtractor, Critic, Classifier, etc.
├── cache/              # Cached data (.pt files)
└── README.md           # This file
```
