from unet.util import simulation
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(
            192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


# use same transform for train/val for this example
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [
                         0.229, 0.224, 0.225])  # imagenet
])

train_set = SimDataset(2000, transform=trans)
val_set = SimDataset(200, transform=trans)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 25

dataloaders = {
    'train': DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0),
    'val': DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(f'dataset_size:{dataset_sizes}')
