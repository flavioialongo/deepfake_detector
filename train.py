import argparse
import torch
import timm
import os
import torchvision.transforms as transforms
from source.attacker import AdversarialAttacker
from source.datasets import AdversarialDataset, RealFakeDataset
from source.evaluate import (plot_pim_training_history, plot_training_history)
from source.trainer import Trainer
from source.configs import Configs
from source.models import SplitModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model with adversarial attacks.")

    parser.add_argument("--train_path", type=str, default="./dffd_small/train", help="Path to the TRAIN dataset")
    parser.add_argument("--test_path", type=str, default="./dffd_small/test", help="Path to the TEST dataset")
    parser.add_argument("--pim", action="store_true", help="Whether to use model with PIM Injection or not")
    parser.add_argument("--adv_train", action="store_true", help="Whether to train using adversarial images")
    return parser.parse_args()
    

def main():
    args = parse_args()
    
    print("Loading data...")
    
    # Dataset / Dataloader creation
    train_dataset = RealFakeDataset(args.train_path, Configs.train_img_augm)
    test_dataset = RealFakeDataset(args.test_path, Configs.test_img_augm)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Configs.batch_size, 
                                            shuffle=False, collate_fn=test_dataset.collate_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Configs.batch_size,
                                             shuffle=True, collate_fn=train_dataset.collate_fn)
    # Base paths
    model_save_path = "./models"
    train_plots_path = "./train_plots"

    # Create base directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(train_plots_path, exist_ok=True)

    print("Building model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)

    for p in model.parameters():
        p.requires_grad = True


    # Initialize attacker
    attacker = AdversarialAttacker(model, 
                    loss=Configs.loss(),
                    mean=Configs.MEAN, 
                    std=Configs.STD, 
                    device=device, 
                    attack_type="pgd", 
                    epsilon=Configs.adv_epsilon,
                    iterative_steps=Configs.attack_iter_steps,
                    deepfool_overshoot=Configs.deepfool_overshoot)
    
    model_attacker = None

    # Handle adversarial training paths
    if args.adv_train:
        print(f"Using Adversarial Train (PGD)...")
        model_save_path = os.path.join(model_save_path, "adversarial_train")
        train_plots_path = os.path.join(train_plots_path, "adversarial_train")
        model_attacker = attacker
    else:
        model_save_path = os.path.join(model_save_path, "normal_train")
        train_plots_path = os.path.join(train_plots_path, "normal_train")

    # Handle PIM/non-PIM paths
    if args.pim:
        model_save_path = os.path.join(model_save_path, "pim")
        train_plots_path = os.path.join(train_plots_path, "pim")
    else:
        model_save_path = os.path.join(model_save_path, "non_pim")
        train_plots_path = os.path.join(train_plots_path, "non_pim")

    # Create final directories
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(train_plots_path, exist_ok=True)
    
    trainer = Trainer(model, train_loader, test_loader, Configs, device)

    if args.pim:
        print("Training model with PIM...")        
        trained_model, history = trainer.train_with_pim(
            epochs=Configs.epochs, 
            alpha=Configs.pim_alpha, 
            r=Configs.pim_r, 
            save_dir=model_save_path, 
            save_name="model.pt", 
            attacker=model_attacker, 
            adv_prob=Configs.adv_train_prob, 
            epsilon_choices=Configs.epsilon_choices
        )
        plot_pim_training_history(history, os.path.join(train_plots_path, "training_metrics.png"))
    else:
        print("Training model...")
        trained_model, history = trainer.train(
            epochs=Configs.epochs, 
            save_dir=model_save_path, 
            save_name="model.pt", 
            attacker=model_attacker, 
            adv_prob=Configs.adv_train_prob, 
            epsilon_choices=Configs.epsilon_choices
        )
        plot_training_history(history, os.path.join(train_plots_path, "training_metrics.png"))
        

if __name__ == "__main__":
    main()
