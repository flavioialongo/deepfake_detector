import torch
from tqdm import tqdm
from source.evaluate import evaluate_model 
import random 
import numpy as np 
def calculate_deltas(grad_mean, grad_sigma, r):

    # Batch-Level perturbation, same for all elements in batch 
    avg_grad_mean = grad_mean.mean(dim=0, keepdim=True)   # [1, C, 1, 1]
    avg_grad_sigma = grad_sigma.mean(dim=0, keepdim=True) # [1, C, 1, 1]

    combined_grad = torch.cat([avg_grad_mean.flatten(), avg_grad_sigma.flatten()])
    total_norm = torch.norm(combined_grad, p=2) + 1e-8

    dmu = r * grad_mean / total_norm
    dsigma = r * grad_sigma / total_norm

    return dmu, dsigma

def calculate_f_prime(features, mu, sigma, dmu, dsigma):


    f_norm = (features - mu) / (sigma + 1e-8)
    f_prime = f_norm * (sigma + dsigma) + (mu + dmu)

    return f_prime

# Training loop with PMI
def train_with_pim(model, 
                   train_dataloader, 
                   eval_loader, 
                   optimizer, 
                   criterion, 
                   device, 
                   scheduler, 
                   num_epochs=5, 
                   alpha=0.1, 
                   r=0.1, 
                   patience=5, 
                   save_path=None,
                   attacker=None,
                   adv_prob = 0.5, 
                   epsilon_choices = [0.001, 0.01, 0.1, 0.3]):
    
    model.to(device)
    
    counter = 0
    best_f1 = 0.0
    best_model = None
    
    history = {
        "train_clean_loss": [],
        "train_perturbed_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_f1": []
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_perturbed_loss = 0.0
        epoch_clean_loss = 0.0

        correct = 0
        total = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()

            if attacker is not None and random.random()<adv_prob:
                
                original_epsilon = attacker.epsilon
                epsilon = random.choice(epsilon_choices)
                attacker.set_epsilon(epsilon)
                # Generate adversarial examples
                
                if attacker.attack_type != "deepfool":
                    adv_images = attacker.attack(images, labels)
                elif attacker.attack_type == "deepfool":
                    adv_images = attacker.attack(images)
                else:
                    raise ValueError(f"Unsupported attack type: {attacker.attack_type}")
                
                # Replace original images with adversarial ones
                images = adv_images.detach()
                attacker.set_epsilon(original_epsilon)

            shallow_layers = model.shallow.layers 
            dmus = [0 for _ in range(len(shallow_layers))]
            dsigmas = [0 for _ in range(len(shallow_layers))]
            means = [0 for _ in range(len(shallow_layers))]
            stds = [0 for _ in range(len(shallow_layers))]
            f_primes = [0 for _ in range(len(shallow_layers))]
            
            for i, layer in enumerate(shallow_layers):
                
                if i == 0:
                    feature = layer(images)
                else:
                    feature = layer(f_primes[i - 1])

                means[i] = torch.mean(feature, [2, 3], keepdim=True)
                stds[i] = torch.std(feature, [2, 3], keepdim=True)

                means[i].retain_grad()
                stds[i].retain_grad()

                # in this stage, f_prime is equal to feature (dmu, dsigma=0)
                # we do that just to start the gradient graph 
                f_primes[i] = calculate_f_prime(feature, means[i], stds[i], dmus[i], dsigmas[i])
                f_primes[i].retain_grad()

            deep = model.deep
            logits = deep(f_primes[-1])
            loss = criterion(logits, labels)
            epoch_clean_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss.backward(retain_graph=True)

            for i in range(len(shallow_layers)):
                dmu, dsigma = calculate_deltas(means[i].grad, stds[i].grad, r)
                dmus[i] = dmu.detach()
                dsigmas[i] = dsigma.detach()

            for i, layer in enumerate(shallow_layers):
                f_primes[i] = calculate_f_prime(f_primes[i], means[i], stds[i], dmus[i], dsigmas[i])

            logits_perturbed = deep(f_primes[-1])
            
            
            loss_perturbed = criterion(logits_perturbed, labels)
            loss_clean = loss
            
            total_loss = (1 - alpha) * loss_clean + alpha * loss_perturbed


            epoch_perturbed_loss += loss_perturbed.item()
            total_loss.backward()
            
            optimizer.step()

        avg_clean_loss = epoch_clean_loss / len(train_dataloader)
        avg_perturbed_loss = epoch_perturbed_loss / len(train_dataloader)
        train_accuracy = 100.0 * correct / total

        test_loss, test_accuracy, test_f1 = evaluate_model(model=model, 
                                                           test_dataloader=eval_loader, 
                                                           criterion=criterion, 
                                                           device=device, 
                                                           visualize_bar=False,
                                                           verbose=False)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Clean Loss: {avg_clean_loss:.4f} | "
            f"Train Perturbed Loss: {avg_perturbed_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.2f}% | "
            f"Test F1: {test_f1:.4f} | "
            f"(Best F1 {best_f1:.4f}) |"
            f"(Early stopping countdown {patience-counter}) |"
            
        )

        # Save metrics to history
        history["train_clean_loss"].append(avg_clean_loss)
        history["train_perturbed_loss"].append(avg_perturbed_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        history["test_f1"].append(test_f1)

        if test_f1 > best_f1:
            best_model = model.state_dict()
            counter = 0
            best_f1 = test_f1
            if save_path:
                torch.save(best_model, save_path)
                print(f"✔️ Saved new best model at epoch {epoch+1} with F1: {test_f1:.4f}")
            if(best_f1>=0.94):
                    print("")
                    break
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered...")
                break
        
        if(scheduler):
            scheduler.step(test_f1)

    model.load_state_dict(best_model)
    return model, history
