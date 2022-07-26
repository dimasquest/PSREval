import torchattacks
import torch 
import utilities as utils 


def run_attack(model, data_loader, name, device='cpu', attack_method=torchattacks.PGD, eps=8/255, alpha=2/255, steps=20):
    atk = attack_method(model, eps=eps, alpha=alpha, steps=steps)
    correct = 0
    images, labels = iter(data_loader).next()
    images, labels = images.to(device), labels.to(device)
    adv_images = atk(images, labels)
    outputs = model(adv_images)

    _, pre = torch.max(outputs.data, 1)

    correct += (pre == labels).sum()

    print(f'Robust accuracy for {name}: %.2f %%' % (100 * float(correct) / len(images)))
    return adv_images, labels, 100 * float(correct) / len(images)


def test_generated_samples(model, labels, adv_images, name, device='cpu'):
    correct = 0
    images, labels = adv_images.to(device), labels.to(device)
    outputs = model(images)
    _, pre = torch.max(outputs.data, 1)
    correct += (pre == labels).sum()
    print(f'Robust accuracy for {name}: %.2f %%' % (100 * float(correct) / len(images)))


def run_simulataneous_attacks(model, data_loader, name, num_classes, device='cpu', eps=8/255, eps_l2=1, alpha=2/255, alpha_l2=0.2, steps=20):
    print(f'Adding an FGSM attack.')
    atk = torchattacks.FGSM(model, eps=eps)
    print(f'Adding a PGD attack.')
    atk2 = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    print(f'Adding a PGD (L2) attack.')
    atk3 = torchattacks.PGDL2(model, eps=eps_l2, alpha=alpha_l2, steps=steps)
    print(f'Adding a FAB attack.')
    atk4 = torchattacks.FAB(model, eps=eps, steps=steps, n_classes=num_classes, n_restarts=3, targeted=False)
    print(f'Combining attacks.')
    atk = torchattacks.MultiAttack([atk, atk2, atk3, atk4])

    correct = 0
    images, labels = iter(data_loader).next()
    images, labels = images.to(device), labels.to(device)
    adv_images = atk(images, labels)
    outputs = model(adv_images)

    _, pre = torch.max(outputs.data, 1)

    correct += (pre == labels).sum()

    print(f'Robust accuracy for {name}: %.2f %%' % (100 * float(correct) / len(images)))
    return adv_images, labels 


def generate_adv_samples(model, images, target, attack_method="PGD", eps=4/255, alpha=2/255, steps=20, num_classes=10):
    atk = utils.select_attack(attack_method, model, eps=eps, alpha=alpha, steps=steps, num_classes=num_classes)
    adv_images = atk(images, target)
    outputs = torch.argmax(model(adv_images).detach().cpu(), axis=1)
    return adv_images, outputs
