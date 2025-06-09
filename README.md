# Deepfake Detection Under Adversarial Attacks

This repository contains the code and experiments for a Computer Vision project developed at Sapienza University. <br>
Our work investigates how the performance of Deepfake Detectors deteriorates when exposed to simple adversarial perturbations applied to input images.

To counter this vulnerability, we further implement and evaluate the Gradient Regularization technique proposed by Guan et al. (2024), introducing a custom Perturbation Injection module to enhance model robustness.

---

# CLI Arguments:
-  ```train_path``` -- Path to the **training** dataset
-  ```test_path``` -- Path to **test** dataset
-  ```val_path``` -- Path to **validation** dataset
-  ```pmi_train``` -- Whether to train using **PMI Injection**

# Example
To run training with PMI Injection:

``` python3 -m main --train_path ./dffd_small/train --test_path ./dffd_small/test --val_path ./dffd_small/validation --pmi_train 1```
