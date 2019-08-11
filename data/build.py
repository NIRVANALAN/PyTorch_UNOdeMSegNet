import torch
from torchvision import transforms
from .Microscopy_Data import MicroscopyDataset


def build_train_loader(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = MicroscopyDataset(args.data.root, args.data.train_list,
                                      args.data.train_img_size, transform=train_transform)
    train_batch_size = args.data.train_batch_size

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size, shuffle=True, num_workers=args.data.workers)
    return train_loader


def build_val_loader(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = MicroscopyDataset(args.data.root, args.data.test_list,
                                    args.data.test_img_size, transform=val_transform)
    test_batch_size = args.data.test_batch_size
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=test_batch_size, shuffle=True,
                                             num_workers=args.data.workers)
    return val_loader

    # use same transform for train/val for this example
# trans = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [
#                          0.229, 0.224, 0.225])  # imagenet
# ])

# train_set = MicroscopyDataset(2000, transform=trans)
# val_set = MicroscopyDataset(200, transform=trans)
