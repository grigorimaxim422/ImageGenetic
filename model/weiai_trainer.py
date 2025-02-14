import torch
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
import math
from torchinfo import summary

from model.simplecnn import SimpleCNN
from model.efficient01A import EfficientNet01A
from model.squeezenet import SqueezeNet
from model.mobilenetv2 import MobileNetV2

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

def get_network(network):
    """ return given network
    """
    if network=='simplecnn':
        net = SimpleCNN()    
    elif network == 'efficient01A':
        net = EfficientNet01A()   
    elif network=='squeezenet':
        net = SqueezeNet()   
    elif network=='mobilenetv2':
        net = MobileNetV2()   
    else:
        net = SimpleCNN()
    
    return net
def check(net):
    logging.info("----------------------------------------------")
    summary(net, input_size=(1, 3, 32, 32), col_names=("input_size", "output_size", "num_params", "mult_adds"))
    logging.info("----------------------------------------------")
    
class WeiaiTrainer:
    def __init__(self, epochs=50, batch_size=128, learning_rate=0.025, momentum=0.9, weight_decay=3e-4, cutout_length=16, network='simplecnn'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cutout_length = cutout_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network=network
        
        self.model = get_network(network)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs,eta_min=0.0001)

        # Data loading and normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        # Adding Cutout to the transform
        transform_train.transforms.append(Cutout(self.cutout_length))
        # CIFAR-100 normalization statistics
        transform_train.transforms.append(transforms.Normalize((0.50707516, 0.48654887, 0.44091785), (0.26733429, 0.25643846, 0.27615047)))

        g = torch.Generator()
        g.manual_seed(0)

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
    
    def train(self, save_path):
        self.set_seed(0)
        self.initialize_weights(self.model)
        best_acc = 0
        for epoch in range(self.epochs):
            self.scheduler.step()
            logging.info(f"Epoch {epoch}, LR: {self.scheduler.get_lr()[0]}")
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take the f
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            acc = self.test()
            if acc > best_acc:
                best_acc = acc
                if acc > 50.:
                    best_acc = acc
                    scripted_model = torch.jit.script(self.model)
                    scripted_model.save(f"{save_path}.best")
                    print(f"model saved into {save_path}.best...")
            
            
            if epoch>0 and epoch % 5 == 1:                
                check(self.model)
                print(f"current best_acc = {best_acc} %")
                

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take the first 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        print(f'({self.network}): Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
        return acc

    def get_model(self):
        return self.model
    
    def reset_model_weights(self, layer, layer_name=''):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            if hasattr(layer, 'children'):
                for name, child in layer.named_children():
                    child_name = f"{layer_name}.{name}" if layer_name else name
                    if isinstance(child, nn.Conv2d):
                        child.reset_parameters()
                    else:
                        self.reset_model_weights(child, child_name)

    def initialize_weights(self,model):
        self.set_seed(0)
            
        # state_dict = model.state_dict()
        # for name, tensor in model.state_dict().items():
        #     if len(tensor.shape) >= 2:  # Ensure the tensor has at least two dimensions
        #         nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        #     elif len(tensor.shape) == 1:  # Handle biases and 1D tensors separately
        #         if 'bias' in name:
        #             nn.init.constant_(tensor, 0)
        #         elif 'weight' in name:
        #             nn.init.constant_(tensor, 1.0)


        for name, param in model.named_parameters():
            if param.dim() >= 2:  # Ensure the parameter has at least two dimensions
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
            elif param.dim() == 1:  # Handle biases separately if they are one-dimensional
                if 'bias' in name:
                    nn.init.constant_(param.data, 0)
                if 'weight' in name:
                    nn.init.constant_(param.data, 1.0) 
                    # nn.init.uniform_(tensor, -0.05, 0.05)
        self.reset_model_weights(model)

