import torch
import os
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
import os
import numpy as np
import torch.backends.cudnn as cudnn
import random
import time
import math
import logging
from selfic.ersten_model import ErstenNet
from selfic.squeezenet import SqueezeNet
from torch.utils.tensorboard import SummaryWriter
from selfic.mobilenetv2 import MobileNetV2
from selfic.utils import *

logging.basicConfig(level=logging.INFO)

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
    
MILESTONES = [60, 120, 160]
class ErstenTrainer:
    def __init__(self, epochs=50, batch_size=128, learning_rate=0.025, momentum=0.9, weight_decay=3e-4, cutout_length=16, net_type="ersten", warm=1, resume=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.b = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cutout_length = cutout_length
        self.resume = resume
        self.warm = warm
        self.net_type = net_type
        self.gpu = True
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model = ErstenNet(num_classes=100).to(self.device)
        if net_type=="ersten":
            self.net = ErstenNet(num_classes=100).to(self.device)
        elif net_type=="squeezenet":
            self.net = SqueezeNet(class_num=100).to(self.device)
        elif net_type=="mobilenetv2":
            self.net = MobileNetV2(num_classes=100).to(self.device)
        else:
            logging.warning(f"No suitable model{net_type}")
            return None
        
        g = torch.Generator()
        g.manual_seed(0)
        
        self.checkpoint_path = "./saved_model"
        
        self.log_path = os.path.join(self.checkpoint_path, '{net}-{epoch}-{type}.log')
        self.checkpoint_path = os.path.join(self.checkpoint_path, '{net}-{epoch}-{type}.pt')

        LOG_DIR='./cache'
        #use tensorboard
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)
        logging.info("ersten_trainer:84")
        #since tensorboard can't overwrite old values
        #so the only way is to create a new tensorboard log
        self.writer = SummaryWriter(log_dir=os.path.join(
                LOG_DIR, datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')))
        input_tensor = torch.Tensor(1, 3, 32, 32)
        if self.gpu:
            input_tensor = input_tensor.cuda()
        
        self.writer.add_graph(self.net, input_tensor)
        
        logging.info("ersten_trainer:95")
        
        transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])        
        
        # # Data loading and normalization
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])
        
        # # Adding Cutout to the transform
        # transform_train.transforms.append(Cutout(self.cutout_length))
        # # CIFAR-100 normalization statistics
        # transform_train.transforms.append(transforms.Normalize((0.50707516, 0.48654887, 0.44091785), (0.26733429, 0.25643846, 0.27615047)))  
              
        self.trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, num_workers=5,
                                      worker_init_fn=self.worker_init_fn, generator=g, pin_memory=True, shuffle=False)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # CIFAR-100 normalization statistics
            transforms.Normalize((0.50707516, 0.48654887, 0.44091785), (0.26733429, 0.25643846, 0.27615047))
        ])
        
        self.testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, num_workers=5,
                                     worker_init_fn=self.worker_init_fn, generator=g, pin_memory=True)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), 
                                   lr=self.learning_rate, 
                                   momentum=self.momentum, 
                                   weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)        
        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=MILESTONES,
                                                              gamma=0.2)         #learning rate decay
        
        self.iter_per_epoch = len(self.trainloader)
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.iter_per_epoch*warm)
        

        
    
    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def set_seed(self, seed=0):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
    
    def save_jit(net, model_path):        
        scripted_model = torch.jit.script(net)
        scripted_model.save(model_path)
        
    def train(self):        
        try:
            best_acc = 0
            print('Ersten::train()')
            for epoch in range(1, self.epochs+1):            
                if self.resume:
                    print('not implemented yet resume...')
                    ## Do it later...
                if epoch > self.warm:
                    self.train_scheduler.step(epoch)
                
                self._train(epoch)
                acc = self.eval_training(epoch)
                
                #start to save best performance model after learning rate decay to 0.01
                if best_acc < acc:
                    weights_path = self.checkpoint_path.format(net=self.net_type,
                                    epoch=epoch,
                                    type='best')
                    print('saving weights file to {}'.format(weights_path))
                    save_jit(self.net, weights_path)
                    best_acc = acc
                    continue
                
                if epoch % 50 == 49:
                    weights_path = self.checkpoint_path.format(net=self.net_type, epoch=epoch, type='')
                    print('saving weights file to {}'.format(weights_path))                        
                    save_jit(self.net, weights_path)           
            
        except Exception as e:
            logging.error(f"An error occured: {e}")
        
            
        
    def eval_training(self, epoch, tb=True):
        
        start = time.time()
        self.net.eval()

        test_loss = 0.0 # cost function error
        correct = 0.0

        for (images, labels) in self.testloader:

            if self.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = self.net(images)
            loss = self.loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        finish = time.time()
        if self.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
            
        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(self.testloader.dataset),
            correct.float() / len(self.testloader.dataset),
            finish - start
        ))
        print()

        #add informations to tensorboard
        if tb:
            self.writer.add_scalar('Test/Average loss', test_loss / len(self.testloader.dataset), epoch)
            self.writer.add_scalar('Test/Accuracy', correct.float() / len(self.testloader.dataset), epoch)

        return correct.float() / len(self.testloader.dataset)


        
    def _train(self, epoch):
        start = time.time()
        self.net.train()
        for batch_index, (images, labels) in enumerate(self.trainloader):

            if self.gpu:
                labels = labels.cuda()
                images = images.cuda()

            self.optimizer.zero_grad()
            outputs = self.net(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            n_iter = (epoch - 1) * len(self.trainloader) + batch_index + 1

            last_layer = list(self.net.children())[-1]
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    self.writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    self.writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                self.optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * self.b + len(images),
                total_samples=len(self.trainloader.dataset)
            ))

            #update training loss for each iteration
            self.writer.add_scalar('Train/loss', loss.item(), n_iter)

            if epoch <= self.warm:
                self.warmup_scheduler.step()

        for name, param in self.net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            self.writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

        finish = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


    def get_model(self):
        return self.net
    