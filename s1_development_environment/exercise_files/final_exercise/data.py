import torch
import os
import glob
import matplotlib.pyplot as plt

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    print(os.getcwd())
    path = "dtu_mlops/data/corruptedmnist/corruptmnist_v1/"
    train_images = []
    train_targets = []
    for i in range(6):
        image_part = torch.load(os.path.join(path, f"train_images_{i}.pt"))
        target_part = torch.load(os.path.join(path, f"train_target_{i}.pt"))
        train_images.append(image_part)
        train_targets.append(target_part)

    train_images = torch.cat(train_images, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    test_images = torch.load(os.path.join(path, f"test_images.pt"))
    test_targets = torch.load(os.path.join(path, f"test_target.pt"))

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_targets = train_targets.long()
    test_targets = test_targets.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_targets)
    test_set = torch.utils.data.TensorDataset(test_images, test_targets)

    return train_set, test_set

if __name__ == "__main__":
    train_dataset, test_dataset = corrupt_mnist()

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)

    # Plot example of image and label
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.title(f"Label: {labels[0]}")
    plt.axis('off')
    plt.show()

    