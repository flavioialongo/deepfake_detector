import dataclasses
from typing import Optional
import torch 
import torchvision.transforms as transforms

@dataclasses.dataclass
class Configs:
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 0.00005
    early_stopping_patience: int = 20
    weight_decay: float = 1e-4
    optimizer = torch.optim.Adam 
    loss = torch.nn.CrossEntropyLoss  

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_img_augm = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    test_img_augm = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    pmi_alpha = 0.5
    pmi_r = 0.1
    