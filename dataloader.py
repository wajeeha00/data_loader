import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels.
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory. Assumes each class
                            has its own folder within the directory.
            transform (callable, optional): Optional transforms to be applied
                                            on an image sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Gather all file paths and labels
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and apply transformations
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=2, transform=None):
    """
    Creates a DataLoader for the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset after every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        transform (callable, optional): Optional transforms to be applied
                                        on an image sample.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = CustomDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    # Example usage
    data_dir = "./data/train"  # Path to your dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader = get_dataloader(data_dir, batch_size=16, transform=transform)

    for images, labels in dataloader:
        print(f"Batch size: {images.size()} | Labels: {labels}")
        break
