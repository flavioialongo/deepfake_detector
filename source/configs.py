import dataclasses
from typing import Optional
import torch 
import torchvision.transforms as transforms

@dataclasses.dataclass
class Configs:
    batch_size: int = 32  # ↑ larger batches improve gradient estimates
    epochs: int = 20
    learning_rate: float = 3e-4  
    early_stopping_patience: int = 5  # ↑ allow time for robust training
    weight_decay: float = 5e-6 

    optimizer = torch.optim.AdamW  # better than Adam for regularization
    loss = torch.nn.CrossEntropyLoss  # consider switching to FocalLoss (see below)

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_img_augm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ], p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    test_img_augm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    pim_alpha = 0.5  # ↑ more weight on prior information (if using PIM)
    pim_r = 1   # ↑ perturbation radius for robustness
    

    epsilon_choices = [0.001, 0.01, 0.1, 0.3]
    adv_train_prob = 0.3
    adv_epsilon = 0.01
    attack_iter_steps = 5
    attack_iter_alpha = 0.001
    attack_iter_epsilon = 0.01
    deepfool_overshoot = 0.2
    
