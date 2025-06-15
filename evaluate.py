import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import f1_score
from collections import Counter
import timm  # assuming you use timm models
from source.configs import Configs
from source.datasets import RealFakeDataset
from source.models import SplitModel
from source.evaluate import evaluate_model, report

import argparse
import torch
from torch.utils.data import DataLoader
import timm
import os

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model weights (.pt)")
    parser.add_argument("--test_path", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--verbose", action="store_true", help="Print detailed batch info during evaluation")
    parser.add_argument("--pim", action="store_true", help="Use PIM trained model")
    parser.add_argument("--plot_dir", type=str, default="./plots/", help="Directory to save evaluation plots")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    val_dataset = RealFakeDataset(args.test_path, Configs.test_img_augm)
    val_loader = DataLoader(val_dataset, batch_size=Configs.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    # Create model
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

    if args.pim:
        model = SplitModel(model)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    criterion = Configs.loss()

    print(f"Evaluating model on dataset with {len(val_dataset)} samples...")

    eval_loss, eval_acc, eval_f1 = evaluate_model(
        model=model,
        test_dataloader=val_loader,
        criterion=criterion,
        device=device,
        visualize_bar=False,
        verbose=args.verbose
    )

     # Extract identifiers from model_path
    model_path_parts = args.model_path.split(os.sep)
    if len(model_path_parts) >= 3:
        train_type = model_path_parts[-3]
        pim_type = model_path_parts[-2]
    else:
        train_type = "unknown_train"
        pim_type = "unknown_pim"

    filename = f"eval_{train_type}_{pim_type}.png"

    # Ensure plot directory exists
    os.makedirs(args.plot_dir, exist_ok=True)

    report(
        model=model,
        test_loader=val_loader,
        loss=criterion,
        device=device,
        save_dir=args.plot_dir,
        filename=filename
    )

    print(f"\nFinal evaluation results:\nLoss: {eval_loss:.4f}\nAccuracy: {eval_acc:.2f}%\nF1 Score: {eval_f1:.4f}")


if __name__ == "__main__":
    main()
