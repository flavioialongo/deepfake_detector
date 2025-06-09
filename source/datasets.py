from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from source.attack import AdversarialAttacker

class RealFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Expecting subfolders 'real/' and 'fake/' inside root_dir
        for label_str, label in [("real", 0), ("fake", 1)]:
            folder = Path(root_dir) / label_str
            if not folder.exists():
                continue
            for file in folder.glob("*.png"):
                self.samples.append((file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


    def collate_fn(self, batch):
        images, labels = zip(*batch)  # unzip list of tuples
        images = torch.stack(images)  # stack image tensors into a single batch tensor
        labels = torch.tensor(labels) # convert labels to a tensor
        return {"image": images, "label":labels }

class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, model, mean, std, attack_type="fgsm", epsilon_range=(0.01, 0.03), 
                 distribution="log_normal", adv_prob=0.5, device='cpu', pgd_steps=None, pgd_alpha=None):
        
        self.dataset = base_dataset
        self.mean = mean
        self.std = std
        self.epsilon_range = epsilon_range
        self.distribution = distribution
        self.adv_prob = adv_prob
        self.device = device
        self.model = model.eval().to(device)


        if attack_type not in ("fgsm", "pgd"):
            raise ValueError("Unknown attack type")

        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.attack_type = attack_type    
        self.attacker = AdversarialAttacker(model, mean, std, device, attack_type, pgd_steps, pgd_alpha)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, label = self.dataset[idx]
        image = image.to(self.device)
        label = torch.as_tensor(label, dtype=torch.long, device=self.device).view(1)

        if torch.rand(1).item() < self.adv_prob:
            image = image.unsqueeze(0)
            image.requires_grad = True
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model.zero_grad()
            loss.backward()
            grad = image.grad.data
            epsilon = self.sample_epsilon().to(self.device)

            if self.attack_type == "fgsm":
                image = self.attacker.attack(image, epsilon, grad)
            else:
                image = self.attacker.attack(image, label, epsilon, self.pgd_alpha, self.pgd_steps)

            image = image.squeeze(0).detach()

        return image.cpu(), label.cpu()

    def sample_epsilon(self):
        if self.distribution == 'uniform':
            return torch.empty(1).uniform_(*self.epsilon_range)
        elif self.distribution == 'log_normal':
            log_min = torch.log(torch.tensor(self.epsilon_range[0]))
            log_max = torch.log(torch.tensor(self.epsilon_range[1]))
            return torch.exp(torch.empty(1).uniform_(log_min, log_max))
        else:
            raise ValueError("Invalid distribution")

    def collate_fn(self, batch):
        images, labels = zip(*batch)  # unzip list of tuples
        images = torch.stack(images)  # stack image tensors into a single batch tensor
        labels = torch.tensor(labels) # convert labels to a tensor
        return {"image": images, "label":labels }
    