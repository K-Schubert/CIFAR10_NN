import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

torch.manual_seed(42)

# CIFAR10 dataset (images and labels) -> CONVERTED TO GRAYSCALE
trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

dataset = CIFAR10(root='../CIFAR10_Logistic_Regression/data/', download=False, transform=trans)
test_dataset = CIFAR10(root='../CIFAR10_Logistic_Regression/data/', download=False, train=False, transform=trans)

val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size=64

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class CIFARModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # output layer
        self.linear3 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear3(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

input_size = 1024
hidden_size = 128
num_classes = 10

model = CIFARModel(input_size, hidden_size=hidden_size, out_size=num_classes)

# GPU/CPU Config

print(torch.cuda.is_available())

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
print(device)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)): # allows to apply function to lists or tuples of tensors
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Allows not to move all the data at once onto the GPU, only batches when required (GPU memory is left free)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device) # yield pauses the execution, not store values in memory, forgets about them once iterated
            # no need to remove batch of data from device, done automatically

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

# Training
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Model (on GPU)
model = CIFARModel(input_size, hidden_size=hidden_size, out_size=num_classes)
print(to_device(model, device))

# Initial Accuracy with random weights
history = [evaluate(model, val_loader)]
print(history)

history += fit(5, 0.5, model, train_loader, val_loader)

history += fit(25, 0.1, model, train_loader, val_loader)

history += fit(15, 0.05, model, train_loader, val_loader)

losses = [x['val_loss'] for x in history]
accuracies = [x['val_acc'] for x in history]

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(losses, '-x')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('Loss vs. No. of epochs');

ax2.plot(accuracies, '-x')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_title('Accuracy vs. No. of epochs');
plt.show()

# Model Test Loss and Accuracy
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
print(result)


# Saving Model Parameters
torch.save(model.state_dict(), 'cifar10-nn-weights.pth')

# Sanity Check
model2 = CIFARModel(input_size, hidden_size=hidden_size, out_size=num_classes)
model2.load_state_dict(torch.load('cifar10-nn-weights.pth'))
model2.state_dict()

test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model2, test_loader)
print(result)

hyper_params = {
	'arch': 'Linear(784, 128)+Linear(128,128)+Linear(128,10)',
	'lr1': 0.5,
	'lr2': 0.1,
    'lr3': 0.05,
	'num_epochs': 45,
	'batch_size': 64
}

metrics = {
	'val_acc': 0.4151,
	'val_loss': 1.6812,
	'test_acc': 0.4128,
	'test_loss': 1.6940.
}

import json

with open('hyper_params.json', 'w') as fp:
    json.dump(hyper_params, fp)

with open('metrics.json', 'w') as fp:
    json.dump(metrics, fp)
'''

