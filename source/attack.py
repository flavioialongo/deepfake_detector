import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os 
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import seaborn as sns

from tqdm import tqdm

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class AdversarialAttacker:

    def __init__(self, model, mean, std, device, attack_type = "fgsm", pgd_steps=None, pgd_alpha=None):
        self.model = model.to(device)
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)
        self.device = device
        self.attack_type = attack_type
        if attack_type == "fgsm":
            self.attack = self.fgsm_attack
        elif attack_type == "pgd":
            self.attack = self.pgd_attack

            if(pgd_steps == None):
                raise Exception("PGD chosen but no steps specified")
            if(pgd_alpha == None):
                 raise Exception("PGD chosen but no pgd_alpha specified")
            self.pgd_steps = pgd_steps
            self.pgd_alpha = pgd_alpha

        else:
            raise Exception("Unknown attack type")
        
    def _denormalize(self, tensor):
        return tensor * self.std + self.mean

    def fgsm_attack(self, images, epsilon, grad):
        images_denorm = self._denormalize(images)
        perturbed = images_denorm + epsilon * grad.sign()
        perturbed = torch.clamp(perturbed, 0, 1)
        return (perturbed - self.mean) / self.std

    def pgd_attack(self, images, labels, epsilon, alpha, steps):
        adv_images = images.clone().detach().to(self.device)
        for _ in range(steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = (images + delta).detach()
        return adv_images

    def evaluate_attack(self, dataloader, save_path=None, save_name="adv_conf_matrix", epsilon=0.01, visualize=False, num_visualize=5):
        
        if(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Evaluation logic here using self.model, self.device, etc.
        self.model.eval()
        self.model.to(self.device)

        correct_adv = 0
        total_samples = 0

        all_labels = []
        all_preds_adv = []

        for batch in tqdm(dataloader):
            images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
            images.requires_grad = True

            self.model.zero_grad()
            # Forward pass on clean images
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)

            # Zero grad and backward pass

            loss.backward()

            # Get gradients
            image_grad = images.grad.data

            if(self.attack_type == "fgsm"):
                adv_images = self.attack(images, epsilon, image_grad)
            else:
                adv_images = self.attack(images, labels, epsilon, self.pgd_alpha, self.pgd_steps)
                
            # Forward pass on adversarial examples
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_preds = adv_outputs.argmax(dim=1)

            # Accuracy
            correct_adv += (adv_preds == labels).sum().item()
            total_samples += labels.size(0)

            # Save predictions and labels for reporting
            all_labels.extend(labels.cpu().numpy())
            all_preds_adv.extend(adv_preds.cpu().numpy())

            if visualize and num_visualize > 0:
                adv_visual = self._denormalize(adv_images, self.mean, self.std, self.device).detach()
                images_visual = self._denormalize(images, self.mean, self.std, self.device).detach()
                for i in range(min(num_visualize, images.size(0))):
                    plt.figure(figsize=(10, 3))

                    orig = images_visual[i].detach().cpu().permute(1, 2, 0).numpy()
                    adv = adv_visual[i].detach().cpu().permute(1, 2, 0).numpy()
                    diff = np.clip((adv - orig) * 10, 0, 1)

                    plt.subplot(1, 3, 1)
                    plt.title(f"Original: {labels[i].item()}")
                    plt.imshow(orig.clip(0, 1))
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.title(f"Adv (ε={epsilon})")
                    plt.imshow(adv.clip(0, 1))
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.title("Difference")
                    plt.imshow(diff)
                    plt.axis('off')

                    plt.show()

                    num_visualize -= 1
                    if num_visualize == 0:
                        break

        accuracy_adv = 100.0 * correct_adv / total_samples if total_samples > 0 else 0.0

        print(f"\nClassification Report on Adversarial Data (epsilon = {epsilon}):")
        print(classification_report(all_labels, all_preds_adv, digits=4))

        conf_matrix = confusion_matrix(all_labels, all_preds_adv)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        file_name = os.path.join(save_path, save_name)
        plt.savefig(file_name)  # or .pdf
        plt.close()

        return accuracy_adv

    def evaluate_epsilons(self, val_loader, epsilons = [0, 0.001, 0.01, 0.1], save_path = None):

        accuracies = []

        for eps in epsilons:
            print(f"\nEvaluating with FGSM (epsilon = {eps})")

            acc = self.evaluate_fgsm(
                model=self.model,
                dataloader=val_loader,   
                device=self.device,
                epsilon=eps,
                visualize=True,        
                num_visualize=2)
            
            accuracies.append(acc)

        self.plot_accuracies(accuracies, epsilons, save_dir=save_path)
        


    def plot_accuracies(self, accuracies, epsilons, save_dir="results/plots", filename="accuracy_vs_epsilon.png"):
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 5))
        plt.plot(epsilons, accuracies, marker='o', linestyle='-', color='blue', label='Adversarial Accuracy')
        
        step = 0.01 if max(epsilons) > 0.05 else 0.005
        plt.xticks(np.arange(0, max(epsilons)+step, step), rotation=45)
        plt.yticks(np.arange(0, 110, step=10))
        plt.title("Model Accuracy vs FGSM Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        print(f"Plot saved to {full_path}")
        plt.close()  # Close the figure to free memory