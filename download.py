import torchvision
import time

def main():
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)        
    testset.download()
    time.sleep(1)    
    
if __name__ == '__main__':      
    main()