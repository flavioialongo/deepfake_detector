import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from source.attacker import AdversarialAttacker
from source.configs import Configs
from source.datasets import RealFakeDataset
import timm

class AttackTester:
    def __init__(self, attacker, dataloader, device, mean, std, output_dir="attack_results", ):
        """
        Initialize attack tester
        
        Args:
            attacker: AdversarialAttacker instance
            dataloader: Test dataloader
            output_dir: Directory to save results
        """

        
        self.attacker = attacker
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.to_pil = ToPILImage()
        self.device=device
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)
        
        os.makedirs(output_dir, exist_ok=True)

    def test_attack(self, num_samples=5):
        """
        Test attack on sample images and save visualizations
        
        Args:
            num_samples: Number of samples to test
        """
      
        # Get sample batch
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= 1:  # Just take first batch
                break
        
            images, labels = batch["image"].to(self.device), batch["label"].to(self.device)

        # Generate adversarial examples
        with torch.no_grad():
            clean_outputs = self.attacker.model(images)
            clean_preds = torch.argmax(clean_outputs, dim=1)
        
        adv_images = self.attacker.attack(images, labels)
        
        
        with torch.no_grad():
            adv_outputs = self.attacker.model(adv_images)
            adv_preds = torch.argmax(adv_outputs, dim=1)
        
        # Save results for first num_samples
        for i in range(min(num_samples, images.size(0))):
            self._save_sample_comparison(
                images[i], 
                adv_images[i],
                labels[i],
                clean_preds[i],
                adv_preds[i],
                idx=i
            )
        
        print(f"Attack test completed. Results saved to {self.output_dir}")

    def _save_sample_comparison(self, clean_img, adv_img, true_label, clean_pred, adv_pred, idx=0):
        """
        Save comparison between clean and adversarial image
        
        Args:
            clean_img: Original clean image tensor
            adv_img: Adversarial image tensor
            true_label: Ground truth label
            clean_pred: Model prediction on clean image
            adv_pred: Model prediction on adversarial image
            idx: Sample index
        """
        # Denormalize images
        clean_img_denorm = self._denormalize(clean_img).detach().squeeze(0).cpu().numpy().transpose(1,2,0)
        adv_img_denorm = self._denormalize(adv_img).detach().squeeze(0).cpu().numpy().transpose(1,2,0)

        # Calculate perturbation
        amplifier = 10
        if(self.attacker.attack_type == "deepfool"):
            amplifier = 50
        
        diff = np.clip((clean_img_denorm - adv_img_denorm) * amplifier, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot images
        titles = [
            f"Original (Pred: {clean_pred}, True: {true_label})",
            f"Adversarial (Pred: {adv_pred})",
            f"Perturbation (x{amplifier})"
        ]
        
        for ax, img, title in zip(axes, [clean_img_denorm, adv_img_denorm, diff], titles):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title)
            ax.axis('off')
        
        # Save figure
        attack_type = self.attacker.attack_type
        plt.suptitle(f"{attack_type} Attack (ε={self.attacker.epsilon})", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{attack_type}_sample_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        

    def _denormalize(self, tensor):
        return tensor * self.std + self.mean

def main(args):

    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)


    model.load_state_dict(torch.load("./models/normal_train/non_pim/model.pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = RealFakeDataset(args.test_path, Configs.test_img_augm)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = Configs.batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

    epsilon = args.epsilon
    attacker = AdversarialAttacker(
        model=model,
        loss=Configs.loss(),
        mean=Configs.MEAN,
        std=Configs.STD,
        device=device,
        attack_type=args.attack_type,
        epsilon=epsilon,
        iterative_steps=Configs.attack_iter_steps,
        deepfool_overshoot=Configs.deepfool_overshoot
    )
    
    output_dir = os.path.join("attack_tests", args.attack_type)

    if(args.attack_type != "deepfool"):
        output_dir = os.path.join(output_dir, f"epsilon_{epsilon}")
    
    # Test attacks
    tester = AttackTester(attacker, test_loader, device, Configs.MEAN, Configs.STD, output_dir)
    tester.test_attack(num_samples=5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["fgsm", "pgd", "deepfool", "ifgsm"],
                        help="Type of attack to test")
    parser.add_argument("--test_path", type=str, default="./dffd_small/validation", help="Test dataset")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon to apply to attack")
    args = parser.parse_args()
    
    main(args)