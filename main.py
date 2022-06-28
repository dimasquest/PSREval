import os
import numpy as np
import torch
import datetime
import train
import quantize
import attack
import utilities as utils
import argparse
import torchattacks
import random
from pathlib import Path
from opacus.validators import ModuleValidator
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device selected: {device}')
path = os.path.dirname(__file__)
time = str(datetime.datetime.now())


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--model', help='Path to trained vanilla model', type=str, default="vanilla_pneu_resnet9.pt")
    parser.add_argument('--dp_model', help='Path to trained DP-SGD model', type=str, default="unwrapped_dpsgd_pneu_dpsgd_resnet9.pt")
    parser.add_argument('--use_trained', help="Specify if a pre-trained model needs to be loaded", action='store_false')
    parser.add_argument('--arch', help='Pick a model arch', default='resnet9', type=str)
    parser.add_argument('--act_func', help='Choose an activation function', default='ReLU', type=str)
    parser.add_argument('--dataset', help='Pick a dataset', default='pneumonia', type=str)

    # dataset arguments
    parser.add_argument('--num_classes', help='Number of classes', type=int, default=3)
    parser.add_argument('--num_channels', help='Number of channels', type=int, default=1)

    # model training arguments
    parser.add_argument('--batch_size_train', help='Batch size (train)', type=int, default=16)
    parser.add_argument('--batch_size_test', help='Batch size (test)', type=int, default=32)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=5)
    parser.add_argument('--dp_epochs', help='Number of epochs for DP', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--adv_training', help='Enable adversarial training', action='store_false')

    # DP arguments
    parser.add_argument('--norm_layer', help='Type of norm layer', type=str, default='group')
    parser.add_argument('--dp_epsilon', help='Value of epsilon for DP (if using make_private_with_epsilon).', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', help='Max gradient norm', type=float, default=1.2)
    parser.add_argument('--delta', help='Value of delta', type=float, default=1e-4)
    parser.add_argument('--noise', help='Noise multiplier', type=float, default=1.1)
    parser.add_argument('--use_epsilon', help='Use a predefined epsilon', action='store_true')

    # attack arguments
    parser.add_argument('--attack', help='Pick the attack method', default="PGD", type=str)
    parser.add_argument('--attack_epsilon', help='Value of epsilon for adversarial samples', type=float, default=8/255)
    parser.add_argument('--adversarial_data', help='Proportion of adversarial data', type=float, default=0.00)
    parser.add_argument('--alpha', help='Value of alphs', type=float, default=2/255)
    parser.add_argument('--steps', help='Number of adversarial steps', type=int, default=20) 
    parser.add_argument('--attack_type', help='Black-box attack type', type=str, default='transfer')

    args = parser.parse_args()

    train_loader, test_loader = utils.load_dataset(args.dataset, args.batch_size_train, args.batch_size_test)
    tags = utils.generate_tags(args.adversarial_data, args.adv_training, args.attack)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    act_func = utils.select_activation_func(args.act_func)
    model_vanilla = utils.load_model(args.arch, in_channels=args.num_channels, num_classes=args.num_classes,
     act_func=act_func, norm_layer=args.norm_layer)
    trained_model = Path(args.model)

    if not trained_model.is_file() or not args.use_trained:
        model_vanilla = train.train_vanilla(
        device=device,
        model=model_vanilla,
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        adv_training=args.adv_training,
        malicious_data=args.adversarial_data,
        attack_method=args.attack,
        eps=args.attack_epsilon,
        alpha=args.alpha,
        steps=args.steps,
        num_classes=args.num_classes
        )
        torch.save(model_vanilla.state_dict(), f"{path}/models/vanilla_{args.dataset}_{args.arch}_{args.adversarial_data}_{tags[0]}_{time}.pt")
    elif args.use_trained:
        print("Model already trained, loading the model.")
        model_vanilla.load_state_dict(torch.load(trained_model), strict=False)

    trained_model_dpsgd = Path(args.dp_model)
    model_dpsgd = utils.load_model(args.arch, in_channels=args.num_channels, num_classes=args.num_classes,
     act_func=act_func, norm_layer=args.norm_layer)
    if not trained_model_dpsgd.is_file() or not args.use_trained:
        model_dpsgd = train.train_dpsgd(
        device=device,
        model=model_dpsgd,
        model_name=args.arch,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        norm_layer=args.norm_layer,
        train_loader=train_loader,
        max_grad_norm=args.max_grad_norm,
        noise_multiplier=args.noise,
        use_epsilon=args.use_epsilon,
        epsilon=args.dp_epsilon,
        delta=args.delta,
        epochs=args.dp_epochs,
        adv_training=args.adv_training,
        learning_rate=args.lr,
        act_func=act_func,
        malicious_data=args.adversarial_data,
        attack_method=args.attack,
        eps=args.attack_epsilon,
        alpha=args.alpha,
        steps=args.steps
        )
        torch.save(model_dpsgd._module.state_dict(), f"{path}/models/dpsgd_{args.dataset}_{args.arch}_{args.adversarial_data}_{tags[0]}_{time}.pt")
        train.remove_hooks(model_dpsgd)
        model_dpsgd = utils.load_model(args.arch, in_channels=args.num_channels, num_classes=args.num_classes,
         act_func=act_func, norm_layer=args.norm_layer).to(device)
        model_dpsgd = ModuleValidator.fix(model_dpsgd)
        model_dpsgd.load_state_dict(torch.load(f"{path}/models/dpsgd_{args.dataset}_{args.arch}_{args.adversarial_data}_{tags[0]}_{time}.pt"))

    elif args.use_trained: 
        print("DP-Model already trained, loading the model.")
        model_dpsgd.load_state_dict(torch.load(trained_model_dpsgd), strict=False)

    print("Comparing vanilla vs quantized:")
    model_quant_vanilla = quantize.quantize_static(model_vanilla, test_loader, "vanilla", f"{path}/models/quant_{args.dataset}_{args.arch}_{args.adversarial_data}_{tags[0]}_{time}.pt")
    
    print("Comparing dpsgd vs quantized:")
    model_quant_dpsgd = quantize.quantize_static(model_dpsgd, test_loader, "dpsgd", f"{path}/models/quant_dpsgd_{args.dataset}_{args.arch}_{args.adversarial_data}_{tags[0]}_{time}.pt")

    print(f'Black-box attacks starting...')
    if args.attack_type == 'transfer':
        print(f'Black-box attack on tranferred images (original model -> target model)')
        adv_images, labels, acc = attack.run_attack(model_vanilla.to('cpu'), test_loader, 'vanilla', attack_method=torchattacks.PGD, 
        eps=args.attack_epsilon, alpha=args.alpha, steps=args.steps)
        attack.test_generated_samples(model_dpsgd.to('cpu'), labels, adv_images, 'dpsgd_from_vanilla', device='cpu')
        attack.test_generated_samples(model_quant_vanilla.to('cpu'), labels, adv_images, 'quantized_vanilla_from_vanilla', device='cpu')
        attack.test_generated_samples(model_quant_dpsgd.to('cpu'), labels, adv_images, 'quantized_dpsgd_from_vanilla', device='cpu')

        print(f'Black-box attack on tranferred images (DP model -> target model)')
        adv_images, labels, acc = attack.run_attack(model_dpsgd.to('cpu'), test_loader, 'dpsgd', attack_method=torchattacks.PGD, 
        eps=args.attack_epsilon, alpha=args.alpha, steps=args.steps)
        attack.test_generated_samples(model_vanilla.to('cpu'), labels, adv_images, 'vanilla_from_dpsgd', device='cpu')
        attack.test_generated_samples(model_quant_vanilla.to('cpu'), labels, adv_images, 'quantized vanilla_from_dpsgd', device='cpu')
        attack.test_generated_samples(model_quant_dpsgd.to('cpu'), labels, adv_images, 'quantized dpsgd_from_dpsgd', device='cpu')

    elif args.attack_type == 'multiattack':
        print(f'Multi-method black-box attack on tranferred images (original model -> target model)')
        adv_images, labels = attack.run_simulataneous_attacks(model_vanilla.to('cpu'), test_loader, 'vanilla', num_classes=args.num_classes,
        eps=args.attack_epsilon, alpha=args.alpha, steps=args.steps)
        attack.test_generated_samples(model_dpsgd.to('cpu'), labels, adv_images, 'dpsgd', device='cpu')
        attack.test_generated_samples(model_quant_vanilla.to('cpu'), labels, adv_images, 'quantized vanilla', device='cpu')
        attack.test_generated_samples(model_quant_dpsgd.to('cpu'), labels, adv_images, 'quantized dpsgd', device='cpu')

        print(f'Multi-method black-box attack on tranferred images (DP model -> target model)')
        adv_images, labels = attack.run_simulataneous_attacks(model_dpsgd.to('cpu'), test_loader, 'dpsgd', num_classes=args.num_classes,
        eps=args.attack_epsilon, alpha=args.alpha, steps=args.steps)
        attack.test_generated_samples(model_vanilla.to('cpu'), labels, adv_images, 'vanilla', device='cpu')
        attack.test_generated_samples(model_quant_vanilla.to('cpu'), labels, adv_images, 'quantized vanilla', device='cpu')
        attack.test_generated_samples(model_quant_dpsgd.to('cpu'), labels, adv_images, 'quantized dpsgd', device='cpu')