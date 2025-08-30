# ==============================================================================
# H2: 5. Dataset Loading Helper --- REPLACE THIS ENTIRE BLOCK ---
# ==============================================================================
def get_dataset(name, root='./data'):
    # Transform for small 32x32 images like CIFAR
    transform_32 = transforms.Compose([
        transforms.ToTensor(),
        # Using standard CIFAR-10 normalization stats as a good default for 32x32
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Transform for larger images, resizing to a standard 224x224
    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Using ImageNet normalization stats, a common default for larger images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Specific transform for EuroSAT which has different channel statistics
    transform_eurosat = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.2034, 0.1367, 0.1148])
    ])

    if name.lower() == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_32)
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_32)
        return train_ds, test_ds, 10, 3

    elif name.lower() == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_32)
        test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_32)
        return train_ds, test_ds, 100, 3
    
    elif name.lower() == 'oxford102flowers':
        train_ds = torchvision.datasets.Flowers102(root=root, split='train', download=True, transform=transform_224)
        test_ds = torchvision.datasets.Flowers102(root=root, split='test', download=True, transform=transform_224)
        return train_ds, test_ds, 102, 3
        
    elif name.lower() == 'eurosat':
        dataset = torchvision.datasets.EuroSAT(root=root, download=True, transform=transform_eurosat)
        # EuroSAT does not have a predefined split, so we create one (80/20).
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
        return train_ds, test_ds, 10, 3
        
    else:
        raise ValueError(f"Unknown or un-configured dataset: {name}")
