# src/dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(
        data_dir, batch_size=64,
        shuffle=True,
        num_workers=2
        ):
    """

    Returns trainloader,testloader, and classnames.
    """

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
        )
    
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
        )

    return train_loader, test_loader, train_dataset.classes