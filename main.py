import argparse
import torch
import timm
import os
import torchvision.transforms as transforms
from source.attack import AdversarialAttacker
from source.datasets import AdversarialDataset, RealFakeDataset
from source.evaluate import (evaluate_model, report)
from source.trainer import Trainer
from source.configs import Configs
from source.models import SplitModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model with adversarial attacks.")

    parser.add_argument("--train_path", type=str, default="./dffd_small/train", help="Path to the TRAIN dataset")
    parser.add_argument("--test_path", type=str, default="./dffd_small/test", help="Path to the TEST dataset")
    parser.add_argument("--val_path", type=str, default="./dffd_small/unseen", help="Path to the VALIDATION dataset")
    parser.add_argument("--pim", type=int, default=0, help="Whether to use model with PIM Injection or not")
    parser.add_argument("--adv_train", type=int, default = 0, help="Whether to train using adversarial images")
    parser.add_argument("--model_path", type=str, default="", help="Path to the model")
    parser.add_argument("--attack_type", type=str, default="fgsm", help = "Attack Type [fgsm, ifgsm, pgd, deepfool]")
    parser.add_argument("--model_type", type=str, default="efficientnet-b2", help="Model architecture to use [efficientnet, resnet]")
    parser.add_argument("--model_eval", type=int, default=0, help="Whether to evaluate the model performances on test / validation dataset")

    return parser.parse_args()
    

def main():
    args = parse_args()
    
    print("Loading data...")
    

    # Dataset / Dataloader creation
    train_dataset = RealFakeDataset(args.train_path, Configs.train_img_augm)
    test_dataset = RealFakeDataset(args.test_path, Configs.test_img_augm)
    val_dataset = RealFakeDataset(args.val_path, Configs.test_img_augm)


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = Configs.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = Configs.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    if(args.attack_type not in ("fgsm", "pgd", "deepfool", "ifgsm")):
        raise Exception("Unknown attack type, choose between fgsm and pgd")
    
    if(args.model_type not in ("efficientnet", "convnext")):
        raise Exception("Unknown attack type, choose between efficientnet and convnext")
    

    model_save_path = "./models"
    
    print("Building model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Plot path is dynamically created based on parameters passed to the cmd line
    plot_save_path = os.path.join("./plots", f"{args.val_path.split("/")[-1]}")
    plot_save_path = os.path.join(plot_save_path, f"{args.attack_type}")



    # Model instantiation
    # If a model path is given, no need to download the pretrained one 
    if(args.model_path):
        if(args.model_type=="efficientnet"):
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        else:
            model = timm.create_model('convnext_tiny', pretrained=False, num_classes=2)
    else:
        if(args.model_type=="efficientnet"):
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        else:
            model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)


    # If model eval == True then we just want to compare the performances on the datasets 
    # And we skip the training / attack 
    if(args.model_eval == True):
        
        if(args.model_path == ""):
            raise Exception("Choose a pretrained model with --model_path")
        
        print(f"Loading model from path {args.model_path}")

        if(args.pim):
            model = SplitModel(model, args.model_type)
        try:
            model.load_state_dict(torch.load(args.model_path))
        except: 
            raise Exception("Error occurred while loading model")
        
        print(f"Test Loader {args.test_path}")
        report(model, test_loader, Configs.loss(), device, None, None)
        
        print(f"Validation loader {args.val_path}")
        report(model, val_loader, Configs.loss(), device, None, None)

    else:

        # Attacker used in Adversarial Train / Evaluation
        attacker = AdversarialAttacker(model, 
                        mean=Configs.MEAN, 
                        std=Configs.STD, 
                        device=device, 
                        attack_type=args.attack_type, 
                        iterative_steps=Configs.attack_iter_steps, 
                        iterative_alpha=Configs.attack_iter_alpha,
                        iterative_epsilon=Configs.attack_iter_epsilon,
                        deepfool_maxiter=Configs.deepfool_maxiter,
                        deepfool_overshoot=Configs.deepfool_overshoot,
                        fgsm_epsilon=Configs.fgsm_epsilon)
        

        # If model path is given, no need to train again, just load it
        if(args.model_path):

            # Model path dynamically changed 
            if(args.pim):
                plot_save_path = os.path.join(plot_save_path, "pim")
            else:
                plot_save_path = os.path.join(plot_save_path, "non_pim")

            print(f"Loading model from path {args.model_path}")
            

            # PIM needs to split the model into Shallow and Deep 
            if(args.pim):
                model = SplitModel(model, args.model_type )
            try:
                model.load_state_dict(torch.load(args.model_path))
            except: 
                raise Exception("Error occurred while loading model")

        else:

            # TRAIN SECTION 
            model_attacker = None

            # TRAIN WITH PIM 
            if(args.pim):
                plot_save_path = os.path.join(plot_save_path, "pim")
                model_save_path = os.path.join(model_save_path, "pim")
                if(args.adv_train == 1):
                    print("Adversarial train not supported with PIM training (memory issues)")
                    return 

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = Configs.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)    
                print("Training model with PIM...")
                trainer = Trainer(model, train_loader, val_loader, Configs, device, model_type = args.model_type)
                trained_model, history = trainer.train_with_pim(Configs.epochs, alpha=Configs.pim_alpha, r=Configs.pim_r, save_dir=model_save_path, save_name=f"{args.model_type}.pt")
            else:

                # TRAIN WITHOUT PIM
                model_save_path = os.path.join(model_save_path, "non_pim/")
                plot_save_path = os.path.join(plot_save_path, "non_pim/")

                if(args.adv_train == 1):
                        print("Using AdversarialDataset...")
                        model_save_path = os.path.join(model_save_path, "adversarial_train/")
                        plot_save_path = os.path.join(plot_save_path, "adversarial_train/")
                        # TRAIN WITH ADVERSARIAL DATASET
                        model_attacker = attacker

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = Configs.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

                print("Training model...")
                trainer = Trainer(model, train_loader, val_loader, Configs, device)
                trained_model, history = trainer.train(Configs.epochs, save_dir=model_save_path, save_name=f"{args.model_type}.pt", attacker=model_attacker, adv_prob=Configs.adv_train_prob, epsilon_choices=Configs.epsilon_choices)
            
            model = trained_model

        plot_save_path = os.path.join(plot_save_path, args.model_type)


        # ATTACK EVALUATION 
        print("Evaluating adversarial robustness...")        
        epsilons = [0.001, 0.01, 0.1, 0.5]
        adv_acc = attacker.evaluate_epsilons(
            val_loader=val_loader,
            epsilons=epsilons,
            save_path=plot_save_path,
            save_name="evaluations"
        )

if __name__ == "__main__":
    main()
