import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm

from functools import partial

def train(device, train_loader, model, criterion, optimizer, lr_scheduler, scaler, accumulation_steps, max_grad_norm, use_amp, epoch):
    model.train()
    
    sum_loss = 0.0
    count = 0
    optimizer.zero_grad()
    
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logit = model(img)
            loss = criterion(logit, label)
            
        scaler.scale(loss).backward()
        
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            # print(f"Accumulation steps: {accumulation_steps}")
            
            # 勾配クリッピング
            scaler.unscale_(optimizer)  # クリッピング前に unscale する
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer の更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
    lr_scheduler.step(epoch)

    return sum_loss, count

def test(device, test_loader, model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            logit = model(img)
            loss = criterion(logit, label)
            
            sum_loss += loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count