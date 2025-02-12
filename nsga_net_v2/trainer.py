import os
import sys
import json
import copy
import asyncio
import logging
import argparse
import numpy as np
from datetime import datetime
from torchinfo import summary
import time
import torch
import torch.nn as nn
import torchvision.utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from nsga_net_v2.autoaugment import CIFAR10Policy
from nsga_net_v2.networks import NSGANetV2
from torchprofile import profile_macs
from nsga_net_v2.evaluate import OFAEvaluator
from validate.forward import *


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--validate_epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.025, help='learning rate')
parser.add_argument('--save_dir', type=str, default="saved_nsgav2/", help="saving path of model ")
parser.add_argument('--data_dir', type=str, default="data/", help="path of dataset loaded")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

    
device = 'cuda'

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class NSGANetV2Trainer:
    def __init__(self, epochs=150, batch_size=128,learning_rate=0.01,
                 momentum=0.9, weight_decay=4e-5, cutout_length=16,
                 grad_clip=5, cutout=True,auto_augment=False,evaluate=True, save=True,
                 topk=10,drop=0.2, drop_path=0.2, img_size=224, save_dir="./saved_nsgav2/", data_dir="./data", num_workers=1, report_freq=50):
        #img_size:192~256
        self.epochs = epochs
        self.batch_size =batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.report_freq = report_freq
        self.cutout_length = cutout_length
        self.grad_clip = grad_clip
        self.cutout = cutout
        self.auto_augment = auto_augment
        self.evaluate = evaluate
        self.topk = topk
        self.drop = drop
        self.drop_path = drop_path
        self.img_size = img_size
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.save = save
        self.num_workers = num_workers
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        train_transform, valid_transform = self._data_transforms()
        
        train_data = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=valid_transform)
        
        self.train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)

        self.valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=200, shuffle=False, pin_memory=True, num_workers=self.num_workers)
    
        net_config = json.load(open("./nsga_net_v2/net.conf"))
        
        self.net = NSGANetV2.build_from_config(net_config, drop_connect_rate=self.drop)
        # init = torch.load('./cifar100', map_location='cpu')['state_dict']
        # self.net.load_state_dict(init)
        NSGANetV2.reset_classifier(self.net, last_channel=self.net.classifier.in_features, n_classes=100)#CIFAR100
        ##calc params and flops
        inputs = torch.randn(1, 3, self.img_size, self.img_size)
        flops = profile_macs(copy.deepcopy(self.net), inputs) / 1e6
        print(f"flops={flops}M")
        params = sum(p.numel() for p in self.net.parameters() if p.requires_grad) / 1e6
        self.net_name = "net_flops@{:.0f}".format(flops)
        logging.info('#params {:.2f}M, #flops {:.0f}M'.format(params, flops))
        
        # self.net = nn.DataParallel(self.net) ## data gpu parallel
        self.net = self.net.to(device)
        
        n_epochs = epochs

        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        
        self.criterion = nn.CrossEntropyLoss().to(device)

        self.optimizer = optim.SGD(parameters,
                          lr=self.learning_rate,
                          momentum=self.momentum,
                          weight_decay=self.weight_decay)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epochs)
        
    def _data_transforms(self):
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
            transforms.RandomHorizontalFlip(),
        ])
        
        if self.auto_augment:
            train_transform.transforms.append(CIFAR10Policy())
            
        train_transform.transforms.append(transforms.ToTensor())
        
        if self.cutout:
            train_transform.transforms.append(Cutout(self.cutout_length)) #16#args.cutout_length
        
        train_transform.transforms.append(transforms.Normalize(norm_mean, norm_std))

        valid_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=3),  # BICUBIC interpolation
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        return train_transform, valid_transform    
    
    def train(self):
        best_acc = 0  # initiate a artificial best accuracy so far
        top_checkpoints = []  # initiate a list to keep track of
        
        for epoch in range(self.epochs):
            logging.info('epoch %d lr %e', epoch, self.scheduler.get_lr()[0])
            self._train(self.train_queue, self.net, self.criterion, self.optimizer)
            _, valid_acc = self.infer(self.valid_queue, self.net, self.criterion)
            
            #checkpoint saving
            if self.save:
                if epoch % 50 == 49:
                    if(len(top_checkpoints) < self.topk):
                        OFAEvaluator.save_net(self.save_dir, self.net, 
                                            self.net_name+'{}.ckpt'.format(epoch))
                        top_checkpoints.append((os.path.join(self.save_dir, self.net_name+'{}.ckpt'.format(epoch)), valid_acc))
                    else:
                        idx = np.argmin([x[1] for x in top_checkpoints])
                        if valid_acc > top_checkpoints[idx][1]:
                            OFAEvaluator.save_net(self.save_dir, self.net, self.net_name + '{}.ckpt'.format(epoch))
                            top_checkpoints.append((os.path.join(self.save_dir, self.net_name+'{}.ckpt'.format(epoch)), valid_acc))
                            # remove the idx
                            os.remove(top_checkpoints[idx][0])
                            top_checkpoints.pop(idx)
                            print(top_checkpoints)
                        
                if valid_acc > best_acc:
                    OFAEvaluator.save_net(self.save_dir, self.net, self.net_name + '.best.pt')                                        
                    best_acc = valid_acc
            
            self.scheduler.step()
        
        OFAEvaluator.save_net_config(self.save_dir, self.net, self.net_name+'.config')
            
    def get_model(self):
        return self.net
    
    def _train(self, train_queue, net, criterion, optimizer):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for step, (inputs, targets) in enumerate(train_queue):
            inputs = F.interpolate(inputs, size=self.img_size, mode='bicubic', align_corners=False)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            optimizer.step()
            
            train_loss+=loss.item()
            _, predicted = outputs.max(1)
            total +=targets.size(0)
            correct +=predicted.eq(targets).sum().item()
            
            if step%self.report_freq == 0:
                logging.info('train %03d %e accu=%f', step, train_loss/total, 100.*correct/total)                
                
        logging.info('train accu = %f', 100.*correct/total)
        
        return train_loss/total, 100.*correct/total
    
    def infer(self, valid_queue, net, criterion)            :
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(valid_queue):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if step%self.report_freq ==0:
                    logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)
                
        acc = 100.*correct/total
        logging.info('valid acc %f', 100. *correct / total)
        
        return test_loss / total, acc
        

def check():
    net_config = json.load(open('./nsga_net_v2/net.conf'))
    net = NSGANetV2.build_from_config(net_config, drop_connect_rate = 0.02)
    summary(net, input_size=(1, 3, 32, 32), col_names=("input_size", "output_size", "num_params", "mult_adds"))

async def main():
        
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    trainer = NSGANetV2Trainer(epochs=args.epochs)
    trainer.train()
    model = trainer.get_model()
    
    params = sum(param.numel() for param in model.parameters())     
    params = round_to_nearest_significant(params, 1)
    macs = calc_flops_onnx(model)
    print(f"A.🖥️ Params: {params/1000}K")    
    print(f"A.🖥️ Flops: {macs/1000000}M")        
    
        
    
if __name__ == '__main__':
    check()
    time.sleep(10)
    asyncio.run(main())
