import time
import torch
import asyncio
import os
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import random

class FineTunedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # Frozen base model
        self.fc = torch.nn.Linear(512, 100)  # Example: New classifier
    
    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        return x


def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
def main():
    print("Convert torchscript model to trainable...")
    scripted_model = torch.jit.load("../saved_model/ne31-01.pt")
    
    trainable_model = scripted_model.eval()
    for param in trainable_model.parameters():
        print(param.shape)
    
    # Create an instance of the new model
    model = FineTunedModel(scripted_model) 
    
    for param in model.base.parameters():
        param.requires_grad = True  # Enable fine-tuning
        
    batch_size = 128    
    epochs = 5
    learning_rate = 0.001
    momentum=0.9
    weight_decay=3e-4
    
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    g = torch.Generator()
    g.manual_seed(0)
        
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=5,
                                      worker_init_fn=worker_init_fn, generator=g, pin_memory=True, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs,eta_min=0.0001)
    
    for epoch in range(5):
        scheduler.step()
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    fine_tuned_scripted = torch.jit.script(model)
    torch.jit.save(fine_tuned_scripted, "../saved_model/fine_tuned_model.pt")

main()