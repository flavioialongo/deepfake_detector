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

    def __init__(self, model, 
                 loss,
                 mean, 
                 std, 
                 device, 
                 epsilon,
                 iterative_steps,
                 deepfool_overshoot,
                 attack_type = "fgsm",
                 ):
        
        self.model = model.to(device)
        self.loss = loss
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)
        self.device = device
        self.attack_type = attack_type
        self.epsilon = epsilon
        
        if attack_type == "fgsm":
            self.attack = self.fgsm_attack
        elif attack_type == "pgd":
            self.attack = self.pgd_attack
        elif attack_type == "ifgsm":
            self.attack = self.i_fgsm_attack
            self.steps = iterative_steps
            self.alpha = 2*epsilon / iterative_steps
    
        elif attack_type == "deepfool":
            self.attack = self.deepfool_attack
            self.deepfool_overshoot = deepfool_overshoot
            self.deepfool_maxiter = iterative_steps
        else:
            raise Exception("Unknown attack type")
        
    
        self.steps = iterative_steps
        self.alpha = 2*epsilon / iterative_steps

    def _denormalize(self, tensor):
        return tensor * self.std + self.mean

    def fgsm_attack(self, images, labels):
        mean = self.mean
        std = self.std
        
        # Denormalize to [0,1]
        x_orig = images * std + mean
        
        # Clone and detach, then enable gradient tracking
        x_adv = x_orig.clone().detach().requires_grad_(True)
        x_adv = x_adv.to(self.device)
        
        # Forward pass (with normalize input)
        x_input = (x_adv - mean) / std
        outputs = self.model(x_input)
        loss = self.loss(outputs, labels)
        
        # Compute gradients w.r.t the image
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        
        # Update adversarial image with sign of gradient
        x_adv = x_adv + self.epsilon * grad.sign()
        
        # Clip to ensure valid pixel range [0,1]
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # Renormalize before returning
        return (x_adv - mean) / std

    def i_fgsm_attack(self, images, labels):

        # Denormalize
        mean = self.mean
        std = self.std
        x_orig = images * std + mean  # [0, 1] range
        model = self.model
        
        # Initialize adversarial image
        x_adv = x_orig.clone().detach()
        x_adv = x_adv.to(self.device)
        
        for _ in range(self.steps):
            x_adv = x_adv.requires_grad_(True)
            
            # Normalize for model input
            x_input = (x_adv - mean) / std
            outputs = model(x_input)
            loss = self.loss(outputs, labels)
            
            # Compute gradients w.r.t. the image
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
            
            # Update adversarial image with sign of gradient
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            
            # Clip to ensure valid pixel range [0,1]
            x_adv = torch.clamp(x_adv, 0, 1)

        # Renormalize before returning
        x_adv_norm = (x_adv - mean) / std
        return x_adv_norm

    def pgd_attack(self, images, labels):

        # Denormalize
        mean = self.mean
        std = self.std
        images_denorm = images * std + mean
        
        model = self.model

        # Initialize adversarial image
        x_adv = images_denorm.clone().detach()
        
        # Random initialization within epsilon ball
        delta = torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(x_adv + delta, 0, 1).detach()
        
        for _ in range(self.steps):
            adv_images = adv_images.requires_grad_(True)
            outputs = model((adv_images - mean) / std)
            loss = self.loss(outputs, labels)
            
            # Calculate gradient w.r.t the image
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False)[0]
            
            # Update adversarial image with sign of gradient
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            # Project back to epsilon-ball
            delta = torch.clamp(adv_images - x_adv, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(x_adv + delta, 0, 1)
        
        # Normalize before returning
        return (adv_images - mean) / std


    def deepfool_attack(self, image):
        
        # TODO fix this 

        mean = self.mean
        std = self.std
        model = self.model 

        image_denorm = image * std + mean
        pert_image = image_denorm.clone().detach()

        r_tot = torch.zeros_like(image).to(self.device)

        loop_i = 0
        with torch.no_grad():
            label = model(image).argmax(dim=1).item()

        while loop_i < self.steps:
            pert_image = pert_image.detach().requires_grad_()
            outputs = model((pert_image - mean) / std)
            fs = outputs.flatten()

            grad_orig = torch.autograd.grad(fs[label], pert_image, retain_graph=True)[0]
            min_dist = float('inf')
            w = None

            num_classes = 2
            for k in range(num_classes):
                if k == label:
                    continue
                grad_k = torch.autograd.grad(fs[k], pert_image, retain_graph=True)[0]
                w_k = grad_k - grad_orig
                f_k = (fs[k] - fs[label]).item()
                dist = abs(f_k) / (w_k.norm() + 1e-8)

                if dist < min_dist:
                    min_dist = dist
                    w = w_k

            ri = (min_dist + 1e-4) * w / (w.norm() + 1e-8)
            r_tot = r_tot + ri
            pert_image = image_denorm + (1 + self.overshoot) * r_tot

        
            with torch.no_grad():
                new_label = model((pert_image - mean) / std).argmax(dim=1).item()
            if new_label != label:
                break

            loop_i += 1

        pert_image = torch.clamp(pert_image, 0, 1)
        adv_norm = (pert_image - mean) / std
        return adv_norm

    def evaluate_attack(self, dataloader, save_path=None, save_name="adv_analysis", visualize=False, num_visualize=5):
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        self.model.eval()
        self.model.to(self.device)
        
        results = {
            'clean_correct': 0,
            'clean_total': 0,
            'adv_correct_from_clean': 0,  # Adversarial accuracy on initially correct samples
            'initially_correct_total': 0,
            'real_to_fake_attacks': 0,    # Real samples successfully attacked to fake
            'fake_to_real_attacks': 0,    # Fake samples successfully attacked to real
            'real_total': 0,
            'fake_total': 0
        }
        
        # For detailed reporting
        all_clean_labels = []
        all_clean_preds = []
        all_adv_labels_from_correct = []  # Only from initially correct samples
        all_adv_preds_from_correct = []
        
        # Class-specific analysis
        all_real_clean_preds = []
        all_real_adv_preds = []
        all_fake_clean_preds = []
        all_fake_adv_preds = []
        
        visualization_count = 0
        
        for batch in tqdm(dataloader, desc="Evaluating adversarial robustness"):

            images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
            batch_size = images.size(0)
            
            # CLEAN EVALUATION 
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_preds = clean_outputs.argmax(dim=1)
            
            # Track clean performance
            clean_correct_mask = (clean_preds == labels)
            results['clean_correct'] += clean_correct_mask.sum().item()
            results['clean_total'] += batch_size
            
            # Save clean predictions
            all_clean_labels.extend(labels.cpu().numpy())
            all_clean_preds.extend(clean_preds.cpu().numpy())
            
            # ADVERSARIAL EVALUATION (Only on correctly classified samples)
            correctly_classified_indices = clean_correct_mask.nonzero(as_tuple=True)[0]
            
            if len(correctly_classified_indices) > 0:
                # Extract correctly classified samples
                correct_images = images[correctly_classified_indices]
                correct_labels = labels[correctly_classified_indices]
                correct_images.requires_grad = True
                
                # Choose correct attack type 
                if self.attack_type != "deepfool":
                    adv_images = self.attack(correct_images, correct_labels)
                else:
                    adv_images = self.attack(correct_images)

                # Evaluate adversarial examples
                with torch.no_grad():
                    adv_outputs = self.model(adv_images)
                    adv_preds = adv_outputs.argmax(dim=1)
                
                # Track adversarial robustness (only for initially correct samples)
                adv_correct_mask = (adv_preds == correct_labels)
                results['adv_correct_from_clean'] += adv_correct_mask.sum().item()
                results['initially_correct_total'] += len(correctly_classified_indices)
                
                # Save for detailed reporting
                all_adv_labels_from_correct.extend(correct_labels.cpu().numpy())
                all_adv_preds_from_correct.extend(adv_preds.cpu().numpy())
                
                # Visualization for correctly classified samples
                if visualize and visualization_count < num_visualize and len(correctly_classified_indices) > 0:
                    self._visualize_adversarial_examples(
                        correct_images, adv_images, correct_labels, adv_preds, 
                        self.epsilon, min(num_visualize - visualization_count, len(correctly_classified_indices))
                    )
                    visualization_count += min(num_visualize - visualization_count, len(correctly_classified_indices))
            
            #  CLASS-SPECIFIC ANALYSIS (All samples)

            # This helps understand bias patterns
            images_for_class_analysis = images.clone()
            images_for_class_analysis.requires_grad = True
            
            self.model.zero_grad()
            if self.attack_type != "deepfool":
                all_adv_images = self.attack(images_for_class_analysis, labels)
            else:
                all_adv_images = self.attack(images_for_class_analysis)

            with torch.no_grad():
                all_adv_outputs = self.model(all_adv_images)
                all_adv_preds = all_adv_outputs.argmax(dim=1)
            
            # Separate by class
            real_mask = (labels == 0)
            fake_mask = (labels == 1)
            
            if real_mask.sum() > 0:
                results['real_total'] += real_mask.sum().item()
                # Count successful attacks: real images classified as fake after perturbation
                real_to_fake = ((labels == 0) & (clean_preds == 0) & (all_adv_preds == 1))
                results['real_to_fake_attacks'] += real_to_fake.sum().item()
                
                all_real_clean_preds.extend(clean_preds[real_mask].cpu().numpy())
                all_real_adv_preds.extend(all_adv_preds[real_mask].cpu().numpy())
            
            if fake_mask.sum() > 0:
                results['fake_total'] += fake_mask.sum().item()
                # Count successful attacks: fake images classified as real after perturbation
                fake_to_real = ((labels == 1) & (clean_preds == 1) & (all_adv_preds == 0))
                results['fake_to_real_attacks'] += fake_to_real.sum().item()
                
                all_fake_clean_preds.extend(clean_preds[fake_mask].cpu().numpy())
                all_fake_adv_preds.extend(all_adv_preds[fake_mask].cpu().numpy())
        
        # COMPUTE METRICS 
        clean_accuracy = 100.0 * results['clean_correct'] / results['clean_total']
        
        # Robustness: How well does the model maintain correct predictions under attack?
        if results['initially_correct_total'] > 0:
            robust_accuracy = 100.0 * results['adv_correct_from_clean'] / results['initially_correct_total']
            attack_success_rate = 100.0 - robust_accuracy
        else:
            robust_accuracy = 0.0
            attack_success_rate = 0.0
        
        # Class-specific attack success rates
        real_attack_success_rate = 100.0 * results['real_to_fake_attacks'] / max(results['real_total'], 1)
        fake_attack_success_rate = 100.0 * results['fake_to_real_attacks'] / max(results['fake_total'], 1)
        
        # Bias analysis
        bias_ratio = results['fake_to_real_attacks'] / max(results['real_to_fake_attacks'], 1)
        
        # REPORTING 
        with open(os.path.join(save_path, f"analysis_epsilon{self.epsilon}.txt"), 'w') as f:
            def write_and_print(line):
                print(line)
                f.write(line + '\n')

            # === REPORTING ===
            write_and_print(f"\n{'='*60}")
            write_and_print(f"ADVERSARIAL EVALUATION RESULTS (ε = {self.epsilon})")
            
            attack_param_str = f"Attack: {self.attack_type}"
            if self.attack_type.lower() == 'fgsm':
                pass  
            elif self.attack_type.lower() == 'pgd':
                attack_param_str += f" | pgd epsilon: {self.epsilon} | pgd alpha: {self.alpha} | pgd steps: {self.steps}"
            elif self.attack_type.lower() == 'ifgsm':
                attack_param_str += f" | ifgsm epsilon: {self.epsilon} | ifgsm alpha: {self.alpha} | ifgsm steps: {self.steps}"
            else:
                attack_param_str += f" | overshoot: {self.deepfool_overshoot} | max_iter: {self.deepfool_maxiter}"

            write_and_print(attack_param_str)

            write_and_print(f"{'='*60}")
            write_and_print(f"Clean Accuracy: {clean_accuracy:.2f}%")
            write_and_print(f"Robust Accuracy (on initially correct): {robust_accuracy:.2f}%")
            write_and_print(f"Attack Success Rate: {attack_success_rate:.2f}%")

            write_and_print(f"\nClass-Specific Attack Analysis:")
            write_and_print(f"  Real→Fake attack success: {real_attack_success_rate:.2f}% ({results['real_to_fake_attacks']}/{results['real_total']})")
            write_and_print(f"  Fake→Real attack success: {fake_attack_success_rate:.2f}% ({results['fake_to_real_attacks']}/{results['fake_total']})")
            write_and_print(f"  Bias Ratio (Fake→Real / Real→Fake): {bias_ratio:.2f}")

            if bias_ratio > 2.0:
                write_and_print(f"  ⚠️  WARNING: Strong bias toward predicting 'Real' class!")
            elif bias_ratio < 0.5:
                write_and_print(f"  ⚠️  WARNING: Strong bias toward predicting 'Fake' class!")

            write_and_print(f"\n--- Clean Performance Report ---")
            clean_report = classification_report(
                all_clean_labels, all_clean_preds, 
                target_names=['Real', 'Fake'], digits=4
            )
            write_and_print(clean_report)
            if len(all_adv_labels_from_correct) > 0:
                write_and_print(f"\n--- Adversarial Robustness Report (Initially Correct Samples Only) ---")
                adv_report = classification_report(
                    all_adv_labels_from_correct, all_adv_preds_from_correct, 
                    target_names=['Real', 'Fake'], 
                    digits=4,
                )
                write_and_print(adv_report)
                
                # === VISUALIZATION ===
                if save_path:
                    self._save_analysis_plots(
                        all_clean_labels, all_clean_preds,
                        all_adv_labels_from_correct, all_adv_preds_from_correct,
                        all_real_clean_preds, all_real_adv_preds,
                        all_fake_clean_preds, all_fake_adv_preds,
                        save_path, save_name, self.epsilon
                    )
        
        return {
            'clean_accuracy': clean_accuracy,
            'robust_accuracy': robust_accuracy,
            'attack_success_rate': attack_success_rate,
            'real_attack_success_rate': real_attack_success_rate,
            'fake_attack_success_rate': fake_attack_success_rate,
            'bias_ratio': bias_ratio,
            'results_dict': results
        }

    def _visualize_adversarial_examples(self, clean_images, adv_images, labels, adv_preds, epsilon, num_to_show):
        """Helper function to visualize adversarial examples"""
        adv_visual = self._denormalize(adv_images, self.mean, self.std, self.device).detach()
        clean_visual = self._denormalize(clean_images, self.mean, self.std, self.device).detach()
        
        for i in range(min(num_to_show, clean_images.size(0))):
            plt.figure(figsize=(12, 4))
            
            orig = clean_visual[i].detach().cpu().permute(1, 2, 0).numpy()
            adv = adv_visual[i].detach().cpu().permute(1, 2, 0).numpy()
            diff = np.clip((adv - orig) * 10, 0, 1)
            
            plt.subplot(1, 4, 1)
            plt.title(f"Original\nTrue: {labels[i].item()}")
            plt.imshow(orig.clip(0, 1))
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.title(f"Adversarial\nPred: {adv_preds[i].item()}")
            plt.imshow(adv.clip(0, 1))
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            plt.title("Difference (×10)")
            plt.imshow(diff)
            plt.axis('off')
            
            plt.subplot(1, 4, 4)
            attack_success = "✓ Success" if labels[i].item() != adv_preds[i].item() else "✗ Failed"
            plt.text(0.5, 0.5, f"Attack: {attack_success}\nε = {epsilon}", 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

    def _save_analysis_plots(self, clean_labels, clean_preds, adv_labels, adv_preds,
                            real_clean_preds, real_adv_preds, fake_clean_preds, fake_adv_preds,
                            save_path, save_name, epsilon):
        """Helper function to save analysis plots"""
        

        conf_matrix_dir = f"confmatrix_epsilon{str(epsilon).replace('.', '_')}"
        os.makedirs(os.path.join(save_path, conf_matrix_dir), exist_ok=True)

        conf_matrix_path = os.path.join(save_path, conf_matrix_dir)

        # 1. Clean vs Adversarial Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Clean confusion matrix
        clean_cm = confusion_matrix(clean_labels, clean_preds)
        sns.heatmap(clean_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        axes[0].set_title("Clean Performance")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        
        # Adversarial confusion matrix (only initially correct)
        if len(adv_labels) > 0:
            adv_cm = confusion_matrix(adv_labels, adv_preds)
            sns.heatmap(adv_cm, annot=True, fmt="d", cmap="Reds", ax=axes[1],
                        xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            axes[1].set_title(f"Adversarial Robustness (ε={epsilon})")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")
        
        plt.tight_layout()
        plt.savefig(os.path.join(conf_matrix_path, f"confusion_matrices.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Class-specific attack analysis
        if len(real_clean_preds) > 0 and len(fake_clean_preds) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Real samples: clean vs adversarial
            real_clean_cm = confusion_matrix([0]*len(real_clean_preds), real_clean_preds, labels=[0,1])
            real_adv_cm = confusion_matrix([0]*len(real_adv_preds), real_adv_preds, labels=[0,1])
            
            sns.heatmap(real_clean_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0,0],
                        xticklabels=['Real', 'Fake'], yticklabels=['True Real'])
            axes[0,0].set_title("Real Images - Clean Predictions")
            
            sns.heatmap(real_adv_cm, annot=True, fmt="d", cmap="Reds", ax=axes[0,1],
                        xticklabels=['Real', 'Fake'], yticklabels=['True Real'])
            axes[0,1].set_title(f"Real Images - Adversarial Predictions (ε={epsilon})")
            
            # Fake samples: clean vs adversarial
            fake_clean_cm = confusion_matrix([1]*len(fake_clean_preds), fake_clean_preds, labels=[0,1])
            fake_adv_cm = confusion_matrix([1]*len(fake_adv_preds), fake_adv_preds, labels=[0,1])
            
            sns.heatmap(fake_clean_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1,0],
                        xticklabels=['Real', 'Fake'], yticklabels=['True Fake'])
            axes[1,0].set_title("Fake Images - Clean Predictions")
            
            sns.heatmap(fake_adv_cm, annot=True, fmt="d", cmap="Reds", ax=axes[1,1],
                        xticklabels=['Real', 'Fake'], yticklabels=['True Fake'])
            axes[1,1].set_title(f"Fake Images - Adversarial Predictions (ε={epsilon})")
            
            plt.tight_layout()
            plt.savefig(os.path.join(conf_matrix_path, f"class_specific_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()

    def evaluate_epsilons(self, val_loader, epsilons = [0, 0.001, 0.01, 0.1], save_path = None, save_name="analysis"):

        accuracies = []
        orig_epsilon = self.epsilon
        for eps in epsilons:
            self.set_epsilon(eps)

            if(self.alpha):
                self.alpha = 2*eps / self.steps

            print(f"Evaluating for ε={eps}")
            acc = self.evaluate_attack(
                dataloader=val_loader,   
                save_path=save_path,
                save_name=save_name,
                visualize=False,        
                num_visualize=2)
            
            accuracies.append(acc)
        self.set_epsilon(orig_epsilon)

        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon 

    def set_attack(self, attack):

        self.attack_type = attack 
        if attack == "fgsm":
            self.attack = self.fgsm_attack
            
        elif attack == "pgd":
            self.attack = self.pgd_attack
        elif attack == "ifgsm":
            self.attack = self.i_fgsm_attack
        elif attack == "deepfool":
            self.attack = self.deepfool_attack
        else:
            raise Exception("Unknown attack type")


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