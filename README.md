# PSREval

Library for training and evaluation of collaboratively trained image analysis models. 

Currently supported:
    - image classification on Paediatric pneumonia prediction and CIFAR-10
    - differentially private training (DP-SGD)
    - adversarial training 
    - post-training model compression 
    - adversarial evaluation (any method from torchattacks is compatible)

N.B to use pneumonia classification dataset, you need to download it first and place in a pneumonia/ folder. We used publicly available data from https://github.com/gkaissis/PriMIA [1]. 

Setup:

    conda env create -f environment.yml
    conda activate quantized_dp

# Example use
Default case (pneumonia prediction, ResNet-9):
    python main.py

Advanced example:

    python main.py --dataset pneumonia --delta 1e-4 --num_channels 1 --num_classes 3 --adversarial_data 0.0 --epochs 20 --dp_epochs 30 --batch_size_train 64 --batch_size_test 256 --arch resnet18 --attack_type transfer --attack FGSM --dp_epsilon 7.0 --use_epsilon 

More details on each parameter can be found in main.py -h or by examining the argparser in main.py

To extend datasets, simply add a method for data loading to data.py; add an alias to utilities.py and 
use the appropriate num_channels and num_classes when running the framework e.g.

    train_dataset = datasets.ImageFolder(str(path) + '/dataset/train', transform=data_transforms)
    test_dataset = datasets.ImageFolder(str(path) + '/dataset/test', transform=data_transforms)

	train_loader = DataLoader(
	    train_dataset,
	    batch_size=batch_size_train,
	    shuffle=True,
	    pin_memory=True
	)

	test_loader = DataLoader(
	    test_dataset,
	    batch_size=batch_size_test,
	    shuffle=False,
	    pin_memory=True
	)
    return train_loader, test_loader

    ...
    if name == 'your_data':
        return data.your_dataset(batch_size_train, batch_size_test)
    
    ...
    python main.py --dataset your_data --num_channels 3 --num_classes 25

Same goes for adding your own models, but these need to be compatible with torch.quantize as per https://pytorch.org/docs/stable/quantization.html

N.B. If you encounter a bug in opacus (specifically poisson_sampler being an illegal keyword), update your opacus version and remove the **kwargs keyword from get_noise_multiplier in lines 60 and 67.

[1] - Kaissis, Georgios, et al. "End-to-end privacy preserving deep learning on multi-institutional medical imaging." Nature Machine Intelligence 3.6 (2021): 473-484.
