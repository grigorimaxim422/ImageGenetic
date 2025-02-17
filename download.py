import torchvision
import torchvision.transforms as transforms
import time

def main():
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            # CIFAR-100 normalization statistics
            transforms.Normalize((0.50707516, 0.48654887, 0.44091785), (0.26733429, 0.25643846, 0.27615047))
        ])
        
    self.testset = torchvision.datasets.CIFAR100(root='./data/cifar-100-python', train=False, download=False, transform=transform_test)        
    self.testset.download()
    time.sleep(1)    
    
if __name__ == '__main__':      
    main()