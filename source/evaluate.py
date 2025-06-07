import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    f1_score,
)
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluate_model(model, test_dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            logits = model(images)  # adjust if not split
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")  # or "weighted" or "micro" depending on your use case

    return avg_loss, accuracy, f1

def report(model, test_loader, loss, device, save_dir="results/reports", filename="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    import matplotlib.pyplot as plt
    import os

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []

    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input, target = batch["image"], batch["label"]
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            l = loss(output, target)
            total_loss += l.item()
            preds = torch.argmax(output, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, digits=4))

    label_names = ["REAL", "FAKE"]

    cm = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    print(f"Confusion matrix saved to {full_path}")
    plt.close()
