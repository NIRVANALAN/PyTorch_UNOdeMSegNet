import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import yaml
import argparse
from easydict import EasyDict
from model import ResNet18UNet, ResNet50UNet
from train import train_model
from data import build_val_loader, build_train_loader
import os
import sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())


def main(args):
    device = torch.device('cuda')
    print(device)

    num_class = args.model.get('num_classes', 7)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    model = ResNet18UNet(num_class).to(device)

    # freeze backbone layers
    # Comment out to finetune further
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=10, gamma=0.1)
    num_epochs = args.get('epochs', 60)

# image_datasets = {
#     'train': train_set, 'val': val_set
# }

# batch_size = 25

# dataloaders = {
#     'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
#     'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
# }

# dataset_sizes = {
#     x: len(image_datasets[x]) for x in image_datasets.keys()
# }

# print(f'dataset_size:{dataset_sizes}')

    train_loader = build_train_loader(args)
    val_loader = build_val_loader(args)
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    torch.cuda.empty_cache()
    model = train_model(model, optimizer_ft, exp_lr_scheduler,
                        device, dataloaders, num_epochs=num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax classification loss")

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--config', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config)
    params.seed = args.seed
    params.local_rank = args.local_rank
    main(params)
