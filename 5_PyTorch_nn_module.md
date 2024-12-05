# PyTorch Neural Network (torch.nn) Module: Comprehensive Guide 🔥

## 1. Core Foundation: nn.Module 🏗️

### Base Class Overview
```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
    def forward(self, x):
        return x
```

### Key Features of nn.Module
| Feature | Description | Example |
|---------|-------------|----------|
| Parameter Management | Automatically tracks learnable parameters | `self.parameters()` |
| Device Movement | Easily move model between CPU/GPU | `model.to(device)` |
| Training Modes | Supports train/eval modes | `model.train()` / `model.eval()` |

## 2. Essential Neural Network Layers 📊

### Linear Layers
```python
# Fully Connected Layer
nn.Linear(in_features=784, out_features=256)
```

### Convolutional Layers
```python
# 2D Convolution
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# 3D Convolution
nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3)
```

### Recurrent Layers
```python
# LSTM Layer
nn.LSTM(input_size=100, hidden_size=256, num_layers=2, batch_first=True)

# GRU Layer
nn.GRU(input_size=100, hidden_size=256, num_layers=2)
```

## 3. Activation Functions ⚡

### Common Activations
| Function | Mathematical Form | Usage |
|----------|------------------|--------|
| ReLU | max(0,x) | `nn.ReLU()` |
| Sigmoid | 1/(1+e^(-x)) | `nn.Sigmoid()` |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | `nn.Tanh()` |

```python
# Example of using activations
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    nn.Sigmoid()
)
```

## 4. Loss Functions 📉

### Popular Loss Functions
```python
# Classification
criterion = nn.CrossEntropyLoss()

# Regression
criterion = nn.MSELoss()

# Binary Classification
criterion = nn.BCELoss()
```

### Usage Example
```python
loss_fn = nn.CrossEntropyLoss()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
```

## 5. Container Modules 📦

### Sequential Container
```python
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.BatchNorm2d(64)
)
```

### ModuleList Container
```python
layers = nn.ModuleList([
    nn.Linear(10, 10) for i in range(5)
])
```

## 6. Regularization Techniques 🎯

### Dropout
```python
nn.Dropout(p=0.5)  # 50% dropout probability
nn.Dropout2d(p=0.2)  # Spatial dropout for CNN
```

### Batch Normalization
```python
# For CNN
nn.BatchNorm2d(num_features=64)

# For Linear layers
nn.BatchNorm1d(num_features=256)
```

## 7. Advanced Components 🚀

### Embedding Layers
```python
# Word embeddings
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)
```

### Attention Mechanisms
```python
# Multi-head attention
attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
```

## 8. Best Practices & Tips 💡

1. **Module Initialization**
```python
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
```

2. **Parameter Management**
```python
# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

3. **Device Management**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

This comprehensive guide covers the essential components of PyTorch's nn module. Each section provides practical examples and implementations that you can use as building blocks for your neural network architectures! 🎉

# PyTorch Optimizers (torch.optim) - Complete Guide 🚀

## Table of Contents 📑
1. [Core Concepts](#1-core-concepts)
2. [Optimizer Types](#2-optimizer-types)
3. [Learning Rate Scheduling](#3-learning-rate-scheduling)
4. [Parameter Management](#4-parameter-management)
5. [Best Practices](#5-best-practices)
6. [Advanced Techniques](#6-advanced-techniques)

---

## 1. Core Concepts 🎯

### 1.1 Basic Optimization Flow
```python
import torch.optim as optim

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
optimizer.zero_grad()  # Clear gradients
loss = criterion(output, target)  # Compute loss
loss.backward()  # Compute gradients
optimizer.step()  # Update parameters
```

### 1.2 Parameter Management
| Component | Description | Access Method |
|-----------|-------------|---------------|
| Parameters | Model weights & biases | `model.parameters()` |
| Gradients | Parameter derivatives | `param.grad` |
| State | Optimizer internal state | `optimizer.state_dict()` |

---

## 2. Optimizer Types 🔄

### 2.1 SGD (Stochastic Gradient Descent)
```python
# Basic SGD
sgd = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum
sgd_momentum = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-4
)
```

### 2.2 Adam (Adaptive Moment Estimation)
```python
adam = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False
)
```

### 2.3 RMSprop
```python
rmsprop = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0,
    momentum=0
)
```

### 2.4 Optimizer Comparison Table
| Optimizer | Advantages | Best Use Cases |
|-----------|------------|----------------|
| SGD | Simple, reliable | General purpose, CNN |
| Adam | Adaptive learning rates | Deep networks, NLP |
| RMSprop | Good for RNNs | Recurrent networks |
| AdaGrad | Adapts per-parameter | Sparse data |

---

## 3. Learning Rate Scheduling 📈

### 3.1 Step Learning Rate
```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)
```

### 3.2 Cosine Annealing
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200,
    eta_min=0
)
```

### 3.3 Custom Learning Rate Schedule
```python
class CustomScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs):
        self.total_epochs = total_epochs
        super(CustomScheduler, self).__init__(optimizer)
        
    def get_lr(self):
        epoch = self.last_epoch
        return [base_lr * (1 - epoch/self.total_epochs)
                for base_lr in self.base_lrs]
```

---

## 4. Parameter Management 🔧

### 4.1 Gradient Clipping
```python
# Clip gradient norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip gradient value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### 4.2 Weight Decay Implementation
```python
class OptimWithDecay:
    def __init__(self, model, lr=0.01, weight_decay=1e-4):
        self.params = model.parameters()
        self.lr = lr
        self.weight_decay = weight_decay
        
    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * (
                        param.grad + self.weight_decay * param.data
                    )
```

---

## 5. Best Practices 💡

### 5.1 Optimizer Selection Guide
```python
def get_optimizer(model, optimizer_name='adam'):
    optimizers = {
        'sgd': optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        ),
        'adam': optim.Adam(
            model.parameters(),
            lr=0.001
        ),
        'rmsprop': optim.RMSprop(
            model.parameters(),
            lr=0.01
        )
    }
    return optimizers[optimizer_name.lower()]
```

### 5.2 Learning Rate Finding
```python
def find_lr(model, train_loader, optimizer, criterion):
    lrs = []
    losses = []
    lr = 1e-7
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.param_groups[0]['lr'] = lr
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        lr *= 1.1
        
        if lr > 10 or loss > 100:
            break
            
    return lrs, losses
```

---

## 6. Advanced Techniques 🔬

### 6.1 Custom Optimizer
```python
class CustomOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(CustomOptimizer, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Custom update rule
                d_p = p.grad
                p.add_(d_p, alpha=-group['lr'])
```

### 6.2 Multi-Optimizer Training
```python
class MultiOptimModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(...)
        self.classifier = nn.Sequential(...)
        
    def get_optimizers(self):
        return {
            'features': optim.SGD(
                self.feature_extractor.parameters(),
                lr=0.01
            ),
            'classifier': optim.Adam(
                self.classifier.parameters(),
                lr=0.001
            )
        }
```

### 6.3 Optimization State Management
```python
def save_optimization_state(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    
def load_optimization_state(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

This comprehensive guide covers all major aspects of PyTorch's optim module, providing both theoretical understanding and practical implementation details. Each section includes code examples and best practices for real-world applications. 🎉