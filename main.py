import argparse
import torch
import timm
import sys 

import torchvision.transforms as transforms
from source.attack import AdversarialAttacker
from source.datasets import AdversarialDataset, RealFakeDataset
from source.evaluate import (evaluate_model, report)
from source.trainer import Trainer
from source.configs import Configs

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model with adversarial attacks.")

    parser.add_argument("--train_path", type=str, default="./dffd_small/train", help="Path to the TRAIN dataset")
    parser.add_argument("--test_path", type=str, default="./dffd_small/test", help="Path to the TEST dataset")
    parser.add_argument("--val_path", type=str, default="./dffd_small/validation", help="Path to the VALIDATION dataset")
    parser.add_argument("--pmi_train", type=int, default=0, help="Whether to train with PMI Injection or not")

    return parser.parse_args()
    

def main():
    args = parse_args()

    print("Loading data...")

    train_img_augm = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize(Configs.MEAN, Configs.STD)
    ]),
    test_img_augm = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(Configs.MEAN, Configs.STD)
    ])


    train_dataset = RealFakeDataset(args.train_path, Configs.train_img_augm)
    test_dataset = RealFakeDataset(args.test_path, Configs.test_img_augm)
    val_dataset = RealFakeDataset(args.val_path, Configs.test_img_augm)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = Configs.batch_size, collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = Configs.batch_size, collate_fn=test_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = Configs.batch_size, collate_fn=val_dataset.collate_fn)

    print("Building model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if(args.pmi_train):
        model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=2)
   
        print("Training model...")
        trainer = Trainer(model, train_loader, val_loader, Configs, device)

        trained_model, history = trainer.train_with_pmi(Configs.epochs, alpha=Configs.pmi_alpha, r=Configs.pmi_r, save_dir="./models/pmi", save_name="model_test.pt")


        print("Evaluating adversarial robustness...")
        
        evaluator = AdversarialAttacker(model, Configs.MEAN, Configs.STD, device, "fgsm")
        epsilon = 0.1
        adv_acc = evaluator.evaluate_attack(
            dataloader=test_loader,
            epsilon=epsilon,
            visualize=False,
            num_visualize=1,
            save_path="plots/adv/",
            save_name="confmatrix_test"
        )
        print(f"Adversarial Accuracy (ε={epsilon}): {adv_acc:.2f}%")

    else:

        model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=2)
   
        print("Training model...")
        trainer = Trainer(model, train_loader, val_loader, Configs, device)

        trained_model, history = trainer.train(Configs.epochs, save_dir="./models/non_pmi", save_name="model_test.pt")


        print("Evaluating adversarial robustness...")
        
        evaluator = AdversarialAttacker(model, Configs.MEAN, Configs.STD, device, "fgsm")


        epsilon = 0.1
        adv_acc = evaluator.evaluate_attack(
            dataloader=test_loader,
            epsilon=epsilon,
            visualize=False,
            num_visualize=1,
            save_path="plots/adv/",
            save_name="confmatrix_test"
        )
        print(f"Adversarial Accuracy (ε={epsilon}): {adv_acc:.2f}%")

if __name__ == "__main__":
    main()
