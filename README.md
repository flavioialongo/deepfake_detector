# 🎭 Deepfake Detection Under Adversarial Attacks

## 🔬 **Project Overview**

This repository contains the code and supporting material for a project developed as part of a Computer Vision project at **Sapienza University of Rome** mainly based on the paper<br> **Improving Robustness of Deepfake Detectors through Gradient Regularization** 
<br>([Guan, Weinan and Wang, Wei and Dong, Jing and Peng, Bo, 2024](https://doi.org/10.1109/TIFS.2024.3396064).)

Our investigation reveals how state-of-the-art deepfake detectors are vulnerable to carefully crafted adversarial perturbations. To address this critical security gap, we implement and evaluate the **Gradient Regularization** technique proposed in the literature. Our results show that, when combined with **Adversarial Training**, this hybrid approach significantly enhances the model’s robustness against such perturbations.

We use the [DFFD dataset](https://cvlab.cse.msu.edu/dffd-dataset.html) as our primary data source. Due to hardware limitations, we utilize a subsample consisting of 4,000 training examples (balanced between real and fake) and 2,000 examples for testing and validation.

---

## 🚀 **Quick Start**

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib sklearn tqdm seaborn
```

### Basic Usage
```bash
# Clone the repository
git clone <repository-url>
cd deepfake_detector_

# Train a robust model with PIM and adversarial training
python3 -m train --train_path ./data/train --test_path ./data/test --pim --adv_train

# Evaluate model performance
python3 -m evaluate --model_path ./models/robust_model --test_path ./data/test --pim --verbose
```

---

## 📁 **Core Components**

### 🏋️ **Training Pipeline**
**`train.py`** - Advanced model training with robustness enhancements

| Parameter | Description | Type |
|-----------|-------------|------|
| `--train_path` | Training dataset directory | `str` |
| `--test_path` | Test dataset directory | `str` |
| `--pim` | Enable Perturbation Injection Module | `flag` |
| `--adv_train` | Enable adversarial training | `flag` |

**Example:**
```bash
python3 -m train --train_path ./dffd_small/train --test_path ./dffd_small/test --pim --adv_train
```

### 📊 **Model Evaluation**
**`evaluate.py`** - Simple model performance analysis

| Parameter | Description | Type |
|-----------|-------------|------|
| `--model_path` | Path to trained model | `str` |
| `--test_path` | Test dataset directory | `str` |
| `--pim` | Model trained with PIM | `flag` |
| `--verbose` | Detailed evaluation output | `flag` |

**Example:**
```bash
python3 -m evaluate --model_path ./models/normal_train/pim/model --test_path ./dffd_small/test --pim --verbose
```

### ⚔️ **Adversarial Attack Testing**
**`attack_tester.py`** - Generate adversarial examples for analysis

| Parameter | Description | Type |
|-----------|-------------|------|
| `--attack_type` | Attack algorithm (e.g., `pgd`, `fgsm`) | `str` |
| `--test_path` | Source images directory | `str` |
| `--epsilon` | Perturbation magnitude | `float` |

**Example:**
```bash
python3 -m attack_tester --attack_type pgd --test_path ./dffd_small/test --epsilon 0.1
```

### 🛡️ **Robustness Assessment**
**`attack_model.py`** - Evaluate model robustness against adversarial attacks

| Parameter | Description | Type |
|-----------|-------------|------|
| `--model_path` | Target model for attack | `str` |
| `--test_path` | Test dataset directory | `str` |
| `--attack_type` | Attack methodology | `str` |
| `--pim` | Model uses PIM architecture | `flag` |
| `--output_dir` | Results output directory | `str` |

**Example:**
```bash
python3 -m attack_model --model_path ./models/normal_train/pim/model --test_path ./dffd_small/test --attack_type pgd --pim --output_dir ./attack_analysis
```

---

## 🔧 **Advanced Configuration**

### Supported Attack Types
- **PGD** (Projected Gradient Descent)
- **FGSM** (Fast Gradient Sign Method)
- **IFGSM** (Iterative Fast Gradient Sign Method)

### Train Techniques
- **Base**: Standard EfficientNet-b0 detector trained without perturbations.
- **PIM-Enhanced**: EfficientNet-b0 augmented with the Perturbation Injection Module (PIM) during training.
- **Adversarial Train**: Robust training incorporating PGD adversarial perturbations with a specified probability (```adv_prob```).
- **Hybrid Train**: Combination of PIM and Adversarial Training..

---

## 📈 **Results & Analysis**

Our experiments indicate that PIM alone does not improve model robustness, as adversarial attacks still maintain nearly a 100% success rate. <br>However, when PIM is combined with adversarial training, the hybrid approach results in a significant increase in robustness against perturbations. <br>While the improvements may appear limited, it is important to note that the experiments were conducted on a relatively small subset of the DFFD dataset.
<br>This constrained data setting likely influenced the overall performance and robustness outcomes.

---

## 🙏 **Acknowledgments**
- Author: Flavio Ialongo
- **Sapienza University of Rome** - Computer Vision Course
- **Guan et al. (2024)** - Gradient Regularization methodology

<br>
Note: While this was conducted as a group project, the implementation, experimentation, and report writing were primarily carried out by the author.

---