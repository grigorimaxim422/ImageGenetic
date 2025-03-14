# CIFAR100 Model Training Project

## 1. Purpose

This project aims to train the optimal TorchScript model that not only maximizes accuracy for a given use-case but also minimizes the number of parameters and the computational cost, measured in Floating Point Operations (FLOPs).

### Dataset

- **CIFAR-100 Dataset** (170MB)
- Contains **600 x 100** images
- **Classes**: 100
- **Total Images**: 60,000

### Required Output

- The trained model should be a **TorchScript model**.

- Accuracy: More than 75%, but less than 90% (to prevent overfitting)

- Parameter Size: Ideally 300K–400K

- FLOPs: Ideally 30M–40M


## 2. Installation

Follow these steps to set up the environment:

```sh
# Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
chmod +x ./*.sh

```

## 3. Train and Validate

### Train a Dummy Model

```
./train_offline.sh
```
### Validate a Pretrained model

```
./validate_offline.sh
```
### Train an NSGA-Net V1 Model(Non-TorchScript)

```
cd nsga_net
chmod +x ./train_nsgav1_cifar100.sh
./train_nsgav1_cifar100.sh
```

### Train an NSGA-Net V2 Model(Non-TorchScript)

```
./train_nsgav2_offline.sh
```


## 4. Goal

 Currently, NSGA-Net and NSGA-Net V2 deliver excellent training results, but it cannot be used to train a TorchScript model. The challenge is to redesign the NSGA-Net or NSGA-Net V2 architecture to produce a TorchScript-compatible model while maintaining a balance at ideal pareto frontier between accuracy, parameter size, and FLOPs.

### Target Metrics:
````
- Accuracy: More than 75%, but less than 90% (to prevent overfitting)

- Parameter Size: Ideally 300K–400K

- FLOPs: Ideally 30M–40M
````