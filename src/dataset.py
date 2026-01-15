# src/dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    data_root="/kaggle/input/fer2013",
    batch_size=64,
    num_workers=2
):
    """
    Returns train_loader, test_loader, and class names.
    """

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transforms
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
