import torch
from sklearn.metrics import (
    f1_score,
)
import os 
from tqdm import tqdm
from source.pim_injector import train_with_pim
from source.models import SplitModel
import random 

class Trainer():

    def __init__(self, model, train_loader, val_loader, config, device="cpu", model_type="efficientnet"):

        self.config = config

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
                
        self.optimizer = self.config.optimizer(model.parameters(), lr=self.config.learning_rate, weight_decay = self.config.weight_decay)
        self.criterion = self.config.loss()
        
        self.device = device
        self.model_type = model_type 


    def train(self, epochs, save_dir=None, save_name = "model.pt", attacker=None, adv_prob = 0.5, epsilon_choices = [0.001, 0.01, 0.1, 0.3]):

        os.makedirs(save_dir, exist_ok=True)

        patience_counter = 0
        best_model = None
        best_f1 = 0.0

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": []
        }

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                self.optimizer.zero_grad()

                if attacker is not None and random.random()<adv_prob:
                    images.requires_grad = True
                    # First forward pass to get loss (used to compute gradients for FGSM/IFGSM)
                    outputs = self.model(images)
                    loss_for_attack = self.criterion(outputs, labels)

                    # Compute gradient w.r.t. images only (do not retain graph)
                    data_grad = torch.autograd.grad(loss_for_attack, images, retain_graph=False, create_graph=False)[0]
                    
                    epsilon = random.choice(epsilon_choices)
                    alpha = epsilon / 4 
                    # Generate adversarial examples
                    if attacker.attack_type == "fgsm":
                        adv_images = attacker.attack(images, epsilon, data_grad)
                    elif attacker.attack_type == "pgd":
                        adv_images = attacker.attack(images, labels, epsilon, alpha, attacker.pgd_steps)
                    elif attacker.attack_type == "ifgsm":
                        adv_images = attacker.attack(images, labels, epsilon, alpha, attacker.ifgsm_steps)
                    elif attacker.attack_type == "deepfool":
                        adv_images = attacker.attack(images, attacker.deepfool_overshoot, attacker.deepfool_maxiter)
                    else:
                        raise ValueError(f"Unsupported attack type: {attacker.attack_type}")

                    # Detach to avoid memory issues and prevent gradient tracking through attack
                    adv_images = adv_images.detach()

                    # Final forward pass for training
                    outputs = self.model(adv_images)
                else:
                    outputs = self.model(images)

                # Final loss and backward
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()


                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct / total

            # Evaluation phase
            self.model.eval()
            eval_loss, eval_acc, eval_f1 = self.evaluate(self.val_loader)

            # Save metrics to history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(eval_loss)
            history["val_acc"].append(eval_acc)
            history["val_f1"].append(eval_f1)

            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {eval_loss:.4f} | Test Acc: {eval_acc:.2f}% | Test F1: {eval_f1:.4f}")

            if eval_f1 > best_f1:
                best_f1 = eval_f1
                best_model = self.model.state_dict()
                patience_counter = 0
                if save_dir:
                    model_file = os.path.join(save_dir, save_name)
                    torch.save(self.model.state_dict(), model_file)
                    print(f"✔️ Saved new best model at epoch {epoch+1} with F1: {eval_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print("Early stopping triggered")
                    break

        # Load best model weights after training loop ends
        if best_model is not None:
            self.model.load_state_dict(best_model)

        return self.model, history

    def train_with_pim(self, epochs, alpha, r, scheduler=None, save_dir=None, save_name="model_pmi.pt"):

        # Split into shallow and deep 
        model = SplitModel(self.model, self.model_type)
    
        for s in model.shallow.parameters():
            s.requires_grad = True
        for s in model.deep.parameters():
            s.requires_grad = False

        if(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, save_name)
    
        return train_with_pim(model, 
                              self.train_loader, 
                              self.val_loader,
                              self.optimizer, 
                              self.criterion, 
                              self.device, 
                              scheduler, 
                              epochs, 
                              alpha, 
                              r, 
                              self.config.early_stopping_patience, 
                              file_path)


    def evaluate(self, val_loader):
        # Evaluation phase
        self.model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                eval_correct += (preds == labels).sum().item()
                eval_total += labels.size(0)

        eval_loss /= len(val_loader)
        eval_acc = 100 * eval_correct / eval_total
        eval_f1 = f1_score(all_labels, all_preds, average="macro")  # or "weighted"

        return eval_loss, eval_acc, eval_f1 