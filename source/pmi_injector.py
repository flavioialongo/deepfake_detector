import torch
from tqdm import tqdm
from source.evaluate import evaluate_model 


def calculate_deltas(grad_mean, grad_sigma, r):

    grad_mean_norm = torch.sqrt((grad_mean ** 2).sum(dim=1, keepdim=True)) + 1e-8
    grad_std_norm = torch.sqrt((grad_sigma ** 2).sum(dim=1, keepdim=True)) + 1e-8

    dmu = r*grad_mean/grad_mean_norm
    dsigma = r*grad_sigma/grad_std_norm

    return dmu, dsigma

def calculate_f_prime(features, mu, sigma, dmu, dsigma):

        f_norm = (features - mu) / (sigma+1e-8)

        f_prime = f_norm*(sigma+dsigma) + (mu+dmu)

        return f_prime

# Training loop with PMI
def train_with_pmi(model, train_dataloader, eval_loader, optimizer, loss_ce, device, scheduler, num_epochs=5, alpha=0.1, r=0.1, patience=5, save_path=None):
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

            shallow = model.shallow
            shallow_layers = [shallow.shallow1, shallow.shallow2, shallow.shallow3, shallow.shallow4, shallow.shallow5]

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
                means[i].retain_grad()

                stds[i] = torch.std(feature, [2, 3], keepdim=True)
                stds[i].retain_grad()

                f_primes[i] = calculate_f_prime(feature, means[i], stds[i], dmus[i], dsigmas[i])
                f_primes[i].retain_grad()

            deep = model.deep
            logits = deep(f_primes[-1])
            loss = loss_ce(logits, labels)
            epoch_clean_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss.backward(retain_graph=True)
            grads_g1 = [p.grad.clone() for p in shallow.parameters()]

            optimizer.zero_grad()
            for i in range(len(shallow_layers)):
                dmu, dsigma = calculate_deltas(means[i].grad, stds[i].grad, r)
                dmus[i] = dmu.detach()
                dsigmas[i] = dsigma.detach()

            for i, layer in enumerate(shallow_layers):
                f_primes[i] = calculate_f_prime(f_primes[i].detach(), means[i], stds[i], dmus[i], dsigmas[i])

            logits_perturbed = deep(f_primes[-1])
            loss_perturbed = loss_ce(logits_perturbed, labels)
            epoch_perturbed_loss += loss_perturbed.item()
            loss_perturbed.backward()

            grads_g2 = [p.grad.clone() for p in shallow.parameters()]
            for p, g1, g2 in zip(shallow.parameters(), grads_g1, grads_g2):
                p.grad = (1 - alpha) * g1 + alpha * g2

            optimizer.step()

            del loss, loss_perturbed, logits, logits_perturbed

        avg_clean_loss = epoch_clean_loss / len(train_dataloader)
        avg_perturbed_loss = epoch_perturbed_loss / len(train_dataloader)
        train_accuracy = 100.0 * correct / total

        test_loss, test_accuracy, test_f1 = evaluate_model(model, eval_loader, loss_ce, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Clean Loss: {avg_clean_loss:.4f} | "
            f"Train Perturbed Loss: {avg_perturbed_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.2f}% | "
            f"Test F1: {test_f1:.4f}"
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
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered...")
                break
        
        if(scheduler):
            scheduler.step(test_f1)

    model.load_state_dict(best_model)
    return model, history
