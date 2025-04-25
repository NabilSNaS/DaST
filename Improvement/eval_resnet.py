from __future__ import print_function
import argparse, random
import torch, torchvision
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
import foolbox as fb
from foolbox.criteria import Misclassification, TargetedMisclassification
from vgg import VGG
from resnet import ResNet50


def parse_args():
    parser = argparse.ArgumentParser(description='DaST/CIFAR-10 Attack Evaluation')
    parser.add_argument('--mode', choices=['baseline', 'dast', 'white'], required=True,
                        help='Which substitute to use: baseline=ResNet-50, dast=DaST-VGG13, white=target itself')
    parser.add_argument('--adv', choices=['FGSM','BIM','PGD','CW'], required=True,
                        help='Attack method')
    parser.add_argument('--targeted', action='store_true', help='Run targeted attack')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dast-model', type=str, default='/work/pi_csc592_uri_edu/hafija_uri/DAST/improvement/saved_model/netD_epoch_0.pth',
                        help='Path to DaST substitute')
    parser.add_argument('--baseline-model', type=str, default='/work/pi_csc592_uri_edu/hafija_uri/DAST/improvement/pretrained/resnet50_cifar10.pth',
                        help='Path to ResNet-50 baseline')
    parser.add_argument('--target-model', type=str, default='/work/pi_csc592_uri_edu/hafija_uri/DAST/improvement/vgg_vgg16_final.pth',
                        help='Path to VGG-16 target')
    return parser.parse_args()


def build_attack(name, targeted):
    if name == 'FGSM':
        return fb.attacks.LinfFastGradientAttack()
    if name == 'BIM':
        return fb.attacks.LinfBasicIterativeAttack(abs_stepsize=2/255, steps=10, random_start=False)
    if name == 'PGD':
        return fb.attacks.LinfPGD(steps=20, abs_stepsize=2/255, random_start=True)
    if name == 'CW':
        return fb.attacks.L2CarliniWagnerAttack(binary_search_steps=9, steps=1000, stepsize=1e-2)
    raise ValueError(f"Unknown attack {name}")


def test_adversary(sub_model, target_model, attack, args, testloader):
    sub_model.eval()
    target_model.eval()
    fmodel = fb.PyTorchModel(sub_model, bounds=(0,1))
    criterion_cls = TargetedMisclassification if args.targeted else Misclassification

    n_success = 0
    n_total   = 0
    total_l2  = 0.0

    for inputs, labels in testloader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        with torch.no_grad():
            orig_preds = target_model(inputs).argmax(dim=1)

        if args.targeted:
            # random target different from original
            tgt = torch.randint(0, 10, labels.shape, device=args.device)
            tgt = torch.where(tgt == orig_preds, (tgt + 1) % 10, tgt)
            mask = (orig_preds != tgt)
            if not mask.any():
                continue
            n_total += mask.sum().item()
            inputs, tgt = inputs[mask], tgt[mask]
            criterion = TargetedMisclassification(tgt)
        else:
            mask = (orig_preds == labels)
            if not mask.any():
                continue
            n_total += mask.sum().item()
            inputs, labels = inputs[mask], labels[mask]
            criterion = Misclassification(labels)

        # craft adversarial examples
        _, advs, l2s = attack(fmodel, inputs, criterion, epsilons=0.07)
        advs = advs.to(args.device)
        total_l2 += float(l2s.to(torch.float32).mean()) * inputs.size(0)

        with torch.no_grad():
            adv_preds = target_model(advs).argmax(dim=1)
        if args.targeted:
            n_success += (adv_preds == tgt).sum().item()
        else:
            n_success += (adv_preds != labels).sum().item()

    success_rate = n_success / n_total * 100
    avg_l2 = total_l2 / n_total
    print(f"{args.adv} {'Targeted' if args.targeted else 'Untargeted'} | \\"
          f"Success: {success_rate:.2f}% (n={n_success}, m={n_total}) | Avg L2: {avg_l2:.4f}")


def main():
    args = parse_args()
    cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        sampler=sp.SubsetRandomSampler(list(range(len(testset)))),
        num_workers=2
    )

    # Load target (black-box) model
    target_model = VGG('VGG16').to(args.device)
    ck = torch.load(args.target_model, map_location=args.device)
    target_model.load_state_dict(ck)
    target_model.eval()

    # Load substitute
    if args.mode == 'white':
        sub_model = target_model
    elif args.mode == 'baseline':
        sub_model = ResNet50().to(args.device)
        sub_model.load_state_dict(torch.load(args.baseline_model, map_location=args.device))
    elif args.mode == 'dast':
        sub_model = VGG('VGG13').to(args.device)
        sub_model.load_state_dict(torch.load(args.dast_model, map_location=args.device))
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    # Build attack
    attack = build_attack(args.adv, args.targeted)

    # Evaluate
    test_adversary(sub_model, target_model, attack, args, testloader)

if __name__ == '__main__':
    main()


#This single script now:  
#- Loads CIFAR‑10 test set (all 10k samples).  
#- Supports `--mode` (baseline, dast, white), `--adv`, and `--targeted`.  
#- Evaluates both non-targeted and targeted success rates and L2 distances.  
