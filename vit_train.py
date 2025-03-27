import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from functools import partial

import trainer

import timm
from timm.models import create_model
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import rand_augment_transform
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.models.layers import trunc_normal_, DropPath
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

device = torch.device("cuda")

train_losses = []
train_accs = []
test_losses = []
test_accs = []

s = time()

def main(args):
    num_epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    img_size = args.img_size
    weight_decay = args.weight_decay
    warmup_t = args.warmup_t
    warmup_lr_init = args.warmup_lr_init
    dataset_name = args.dataset
    accumulation_steps = args.accumulation_steps
    max_grad_norm = args.accumulation_steps
    use_amp = args.amp

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])

    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=test_transform, download=False)
        class_names = train_dataset.classes
        criterion = torch.nn.CrossEntropyLoss()

    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100("./data", train=False, transform=test_transform, download=False)
        class_names = train_dataset.classes        
        criterion = torch.nn.CrossEntropyLoss()

    print(class_names)
    print('Class:', len(class_names))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    model = create_model("vit_tiny_patch16_224", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("vit_small_patch16_224", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("vit_base_patch16_224", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("vit_large_patch16_224", pretrained=True, num_classes=len(class_names)) 
    model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = CosineLRScheduler(optimizer=optimizer, t_initial=num_epoch, 
        warmup_t=warmup_t, warmup_lr_init=warmup_lr_init, warmup_prefix=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epoch):
        train_loss, train_count = trainer.train(device, train_loader, model, criterion, optimizer, lr_scheduler, 
                                                scaler, accumulation_steps, max_grad_norm, use_amp, epoch)
        test_loss, test_count = trainer.test(device, test_loader, model)

        train_loss = (train_loss/len(train_loader))
        train_acc = (train_count/len(train_loader.dataset))
        test_loss = (test_loss/len(test_loader))
        test_acc = (test_count/len(test_loader.dataset))

        print(f"epoch: {epoch+1}, train loss: {train_loss}, train accuracy: {train_acc}, test loss: {test_loss}, test accuracy: {test_acc}")


        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    e = time()
    print('Elapsed time is ',e-s)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=int, default=0.05)
    parser.add_argument("--warmup_t", type=int, default=5)
    parser.add_argument("--warmup_lr_init", type=int, default=1e-5)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument('--amp', action='store_true')
    args=parser.parse_args()
    main(args)
