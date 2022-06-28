import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchattacks
import utilities as utils
import attack
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
from collections import OrderedDict


def train_vanilla(device, model, train_loader, epochs, learning_rate, adv_training=False,
 malicious_data=0.05, attack_method='PGD', eps=4/255, alpha=2/255, steps=10, num_classes=10):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    malicious = len(train_loader) * (1 - malicious_data) 
    model.train()

    for epoch in range(epochs):
        losses = []
        accuracies = []

        for i, (images, target) in enumerate(tqdm(train_loader)): 
            optimizer.zero_grad()
            data = images.to(device)
            target = target.to(device)
            if i >= malicious:
                if adv_training:
                    data, _ = attack.generate_adv_samples(model, images, target, attack_method, eps, alpha, steps, num_classes)   
                else:
                    data, target = attack.generate_adv_samples(model, images, target, attack_method, eps, alpha, steps, num_classes)
                data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            if i < malicious:
                preds = torch.argmax(output.detach().cpu(), axis=1)
                labels = target.detach().cpu()
                acc = np.mean(preds.numpy() == labels.numpy())
                losses.append(loss.item())
                accuracies.append(acc)

            loss.backward()
            optimizer.step()
        print(
            f"Train Epoch: {epoch + 1} "
            f"Loss: {np.mean(losses):.6f} "
            f"Acc: {np.mean(accuracies) * 100:.6f} "
        )

    return model


def train_dpsgd(device, model, model_name, num_channels, num_classes, norm_layer, train_loader,
 max_grad_norm, noise_multiplier, use_epsilon, epsilon, delta, epochs, learning_rate, act_func=torch.nn.ReLU, adv_training=False,
 malicious_data=0.05, attack_method=torchattacks.PGD, eps=4/255, alpha=2/255, steps=10):
    model = ModuleValidator.fix(model)
    malicious = len(train_loader) * (1 - malicious_data)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate) 
    privacy_engine = PrivacyEngine()

    if use_epsilon:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True
        ) 
    else:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True
        )

    model.train()

    for epoch in range(epochs):
        losses = []
        accuracies = []

        for i, (images, target) in enumerate(tqdm(train_loader)):   
            optimizer.zero_grad()
            data = images.to(device)
            target = target.to(device)
            if i >= malicious:
                # for now, much simplified
                wrapped_model = utils.load_model(model_name, in_channels=num_channels, num_classes=num_classes,
                 act_func=act_func, norm_layer=norm_layer)
                wrapped_model = ModuleValidator.fix(wrapped_model)
                wrapped_model.load_state_dict(model._module.state_dict())
                if adv_training:
                    data, _ = attack.generate_adv_samples(wrapped_model, images, target, attack_method, eps, alpha, steps, num_classes)
                else:
                    data, target = attack.generate_adv_samples(wrapped_model, images, target, attack_method, eps, alpha, steps, num_classes)
                data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            if i < malicious:
                preds = torch.argmax(output.detach().cpu(), axis=1)
                labels = target.detach().cpu()
                acc = np.mean(preds.numpy() == labels.numpy())
                losses.append(loss.item())
                accuracies.append(acc)

            loss.backward()
            optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta)

        print(
            f"Train Epoch: {epoch + 1} "
            f"Loss: {np.mean(losses):.6f} "
            f"Acc: {np.mean(accuracies) * 100:.6f} "
            f"(ε = {epsilon:.2f}, δ = {delta})"
        )

    return model


def evaluate(model, test_loader, device, name, unused_batches=0):
    model.eval()

    correct = 0
    cnt = 0
    pred_list = []
    with torch.no_grad():
        
        for images, target in test_loader:
            images = images.to(device)
            target = target

            output = model(images).detach().cpu()
            preds = np.argmax(output, axis=1).numpy()
            labels = target.numpy()

            correct += (preds == labels).sum()

            pred_list.append(preds)
            
            cnt += 1
            
            if cnt >= len(test_loader) - unused_batches:
                break

    acc = correct / (len(test_loader.dataset) * cnt / len(test_loader)) * 100
    print(f"Test Accuracy: {acc:.6f}")

    return np.concatenate(pred_list)


def remove_hooks(model):
    model._backward_hooks = OrderedDict()
    model._forward_hooks = OrderedDict()
    model._forward_pre_hooks = OrderedDict()
    for child in model.children():
        remove_hooks(child)
