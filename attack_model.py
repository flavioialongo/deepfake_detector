import argparse
import torch
import os
from torch.utils.data import DataLoader
from source.datasets import RealFakeDataset
from source.attacker import AdversarialAttacker
from source.configs import Configs
from source.models import SplitModel
import timm
import matplotlib.pyplot as plt
import numpy as np
from source.evaluate import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Test model robustness against adversarial attacks.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_path", type=str, default="./dffd_dataset/test", help="Path to the test dataset")
    parser.add_argument("--attack_type", type=str, required=True, 
                       choices=["fgsm", "pgd", "deepfool", "ifgsm"], 
                       help="Type of attack to test")
    
    parser.add_argument("--pim", action="store_true",
                       help="Whether model uses PIM Injection (1) or not (0)")
    
    parser.add_argument("--output_dir", type=str, default="./attack_results", 
                       help="Directory to save robustness evaluation results")
    
    return parser.parse_args()

def load_model(args, device):
    """Load the trained model from checkpoint"""
    
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

    try:
        if(args.pim):
            model = SplitModel(model)
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {args.model_path}: {str(e)}")

def create_attacker(model, args, device):
    """Create adversarial attacker instance"""
    return AdversarialAttacker(
        model=model,
        loss=Configs.loss(),
        mean=Configs.MEAN,
        std=Configs.STD,
        device=device,
        attack_type=args.attack_type,
        epsilon=Configs.adv_epsilon,
        iterative_steps=Configs.attack_iter_steps,
        deepfool_overshoot=Configs.deepfool_overshoot
    )

def evaluate_robustness(attacker, loader, epsilons, output_dir):
    """Evaluate model robustness across different epsilon values"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate clean accuracy first
    _, clean_accuracy, clean_f1 = evaluate_model(attacker.model, loader, attacker.loss, attacker.device, visualize_bar=True, verbose=False)

    print(f"Clean accuracy: {clean_accuracy / 100:.2%} | Clean f1: {clean_f1:.4}")
    
    # Evaluate across different attack strengths
    attacker.evaluate_epsilons(
        val_loader=loader,
        epsilons=epsilons,
        save_path=output_dir,
        save_name=f"robustness_{attacker.attack_type}"
    )
    
    print(f"Robustness evaluation saved to {output_dir}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = args.output_dir
    # Create output directory structure
    if("adversarial" in args.model_path):
        os.makedirs(os.path.join(output_dir, "adversarial_train"), exist_ok=True)
        output_dir = os.path.join(output_dir, "adversarial_train")
    else:
        os.makedirs(os.path.join(output_dir, "normal_train"), exist_ok=True)
        output_dir = os.path.join(output_dir, "normal_train")


    if(args.pim):
        output_dir = os.path.join(output_dir, "pim")
    else:
        output_dir = os.path.join(output_dir, "non_pim")
    
    output_dir = os.path.join(output_dir, f"{args.attack_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading efficientnet model from {args.model_path}...")
    model = load_model(args, device)
    
    # Create attacker
    attacker = create_attacker(model, args, device)
    
    # Prepare datasets
    test_dataset = RealFakeDataset(args.test_path, Configs.test_img_augm)
    test_loader = DataLoader(test_dataset, batch_size=Configs.batch_size, 
                           shuffle=False, collate_fn=test_dataset.collate_fn)
    
    # Test different attack strengths
    epsilons = [0.001, 0.01, 0.1, 0.5]
    
    print("\n=== Evaluating on Test Set ===")
    evaluate_robustness(attacker, test_loader, epsilons, 
                       output_dir)

if __name__ == "__main__":
    main()