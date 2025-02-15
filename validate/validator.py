
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import logging
import time
import numpy as np
import torch.backends.cudnn as cudnn
import random
import argparse
import os
import math
import json
import asyncio
import sys
import logging
from validate.forward import *

logging.basicConfig(level=logging.INFO)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join("cache", 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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

class OfflineValiTrainer:
    def __init__(self, epochs=5, 
                 batch_size=128,
                 learning_rate=0.025,
                 momentum=0.9,
                 weight_decay=3e-4,
                 cutout_length=16,
                 grad_clip = 5,
                 refer="null"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cutout_length = cutout_length
        self.grad_clip = grad_clip
        self.refer = refer
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        

        print(f"device={self.device}")
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

    def train(self, model):
        self.set_seed(0)        
        model = model.to(self.device)        
        model.cuda()
        
        criterion = nn.CrossEntropyLoss()        
        parameters = filter(lambda p: p.requires_grad, model.parameters()) ## added this 
        optimizer = optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        logging.info(f"Initial learning rate is set to {self.learning_rate}")
        with torch.autograd.set_detect_anomaly(False):
            for epoch in range(self.epochs):
                scheduler.step()
                # logging.info(f"Epoch {epoch}, LR: {scheduler.get_lr()[0]}")
               # model.droprate = 0.0 * epoch / self.epochs
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for i, data in enumerate(self.trainloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # inputs, labels = inputs.cuda(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take the first element if the output is a tuple
                    loss = criterion(outputs, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if i % 100 == 99:  # Print every 100 mini-batches
                        accuracy = 100 * correct / total
                        logging.info(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}, accuracy: {accuracy:.2f}%')
                        running_loss = 0.0
                        correct = 0
                        total = 0
                        
                    # if (i+1) % 10 ==0:
                    #     scripted_model = torch.jit.script(model)
                    #     scripted_model.save(save_path)
                    #     print(f"model saved into {save_path}")
                # Test the model after each epoch
                test_accuracy = self.test(model)
         
        return model

    def test(self, model):
        # self.set_seed(0)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take the first 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        logging.info(f'({self.refer}) Accuracy of the network on the 10000 test images: {100 * correct / total}%')
        return 100 * correct / total
    
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Validating')

parser.add_argument('--validate_epochs', type=int, default=50, help='num of training epochs to test weight model and NAS algorithm')
parser.add_argument('--learning_rate', type=float, default=0.025, help="learning rate")
parser.add_argument('--model_path', type=str, default="saved_model/", help="path of saved torchscript model")
parser.add_argument('--net_name', type=str, default='dummy', help='learning rate')
args = parser.parse_args()

def validatoe(model_path="saved_model/model.pt"):
    accuracy = 0
    try:    
        logging.info("validatoe %s...", model_path)
        if not os.path.exists("cache"):
            os.makedirs("cache")            
        
        if os.path.isfile(f"{model_path}-profile{str(args.validate_epochs).zfill(3)}.json"):
            return model_path, 0
            
        model = torch.jit.load(model_path)
        print(f"model loaded from {model_path}")
        params = sum(param.numel() for param in model.parameters())
        params = round_to_nearest_significant(params, 1)
        time.sleep(1)
        
        trainer = OfflineValiTrainer(epochs=args.validate_epochs, learning_rate=args.learning_rate, refer=model_path)    
        trainer.initialize_weights(model)
        retrained_model = trainer.train(model)
        accuracy = math.floor(trainer.test(retrained_model))            
        retrained_params = sum(param.numel() for param in retrained_model.parameters())     
        retrained_params = round_to_nearest_significant(retrained_params, 1)
        retrained_macs = calc_flops_onnx(retrained_model, f"{model_path}",args.validate_epochs)
        
         # Save to a JSON file
        with open(f"{model_path}-profile{str(args.validate_epochs).zfill(3)}.json", "w") as f:
            json.dump({
                "params":params,
                "flops":retrained_macs,
                "accuracy":accuracy
                       }, f, indent=4) 
        
        return model_path, accuracy
    except Exception as e:
        logging.error(f"O-Failed to advertise model on the chain: {e}")        
        return model_path, accuracy
    

def find_model_files(model_dir):
    """Find all model files in the given directory recursively."""
    model_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pt") or file.endswith(".pth"):  
                model_files.append(os.path.join(root, file))
                
    model_files.sort(reverse=True)

    return model_files
    
def main(args):
    ## args.model_path : directory of stored pretrained models
    try:                       
        model_files = find_model_files(args.model_path)
        results = []
        print(f"{model_files}")
        model_files.sort(key=lambda f: os.path.basename(f), reverse=True)
        
        for model_file in model_files:
            validatoe(model_file)            
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     results = pool.map(validatoe, model_files)
       
        # if results:
        #     largest = max(results, key = lambda x: x[1])
        #     largest_model_path, largest_model_size = largest

        #     print(f"Largest file:{largest_model_path}, largest_size={largest_model_size}")
    except Exception as e:
        logging.error(f"O-Failed on validator: {e}", exc_info=True)
                
        
if __name__ == '__main__':
    main(args)
    # validatoe("saved_model/first.pt")
    # asyncio.run(main(args))