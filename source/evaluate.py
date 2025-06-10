import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    f1_score,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_model(model, test_dataloader, criterion, device, visualize_bar=False, verbose=False):

    model.eval()
    eval_loss = 0.0
    eval_correct = 0
    eval_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in (tqdm(test_dataloader) if visualize_bar else test_dataloader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            logits = model(images) 
            loss = criterion(logits, labels)
            eval_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            eval_correct += (preds == labels).sum().item()
            eval_total += labels.size(0)

            if verbose:
                    print(f"Logits: {logits[:5]}")
                    print(f"Predictions: {preds[:10]}")
                    print(f"Labels: {labels[:10]}")
                    print(f"Predicted class distribution: {Counter(all_preds)}")
                    print(f"Label class distribution: {Counter(all_labels)}")
    eval_loss /= len(test_dataloader)
    eval_acc = 100 * eval_correct / eval_total
    eval_f1 = f1_score(all_labels, all_preds, average="macro")  # or "weighted"

    if verbose:
        print(f"Eval Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.2f}%, F1 Score: {eval_f1:.4f}")

    return eval_loss, eval_acc, eval_f1

def report(model, test_loader, loss, device, save_dir="results/reports", filename="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    import matplotlib.pyplot as plt
    import os

    if(save_dir):
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

    if(save_dir):
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        print(f"Confusion matrix saved to {full_path}")
    plt.close()



def plot_training_history(history, save_path=None):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 5))
    
    # 1. Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    if 'val_f1' in history:
        plt.plot(history['val_f1'], label='Validation F1', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()

    # 3. F1 score plot
    if 'test_f1' in history:
        plt.plot(history['test_f1'], label='Test F1', color='green')
        plt.title('Test F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


def plot_pim_training_history(history, save_path=None):
    """Plot adversarial training metrics with clean/perturbed separation"""
    plt.figure(figsize=(18, 6))
    
    # 1. Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_clean_loss'], label='Clean Loss')
    plt.plot(history['train_perturbed_loss'], label='Perturbed Loss')
    plt.plot(history['test_loss'], label='Test Loss', linestyle='--')
    plt.title('Training and Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['test_accuracy'], label='Test Accuracy')
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 3. F1 score plot
    plt.subplot(1, 3, 3)
    if 'test_f1' in history:
        plt.plot(history['test_f1'], label='Test F1', color='green')
        plt.title('Test F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

