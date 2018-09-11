# reference:
# 1. https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py

import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from network import get_mobilenet, get_mobilenet_v2, get_shufflenet

# choose network --- choose 0~2
model_name = ['mobilenet_v1', 'mobilenet_v2', 'shufflenet'][0]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001

# Image preprocessing modules
# Note: due to "scale=32", here resize image to 64
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Pad(8),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64),
    transforms.ToTensor()])

transform_target = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()])

root = os.path.expanduser('~')
# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(root, 'data'),
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(root, 'data'),
                                            train=False,
                                            transform=transform_target)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

if model_name == 'mobilenet_v1':
    model = get_mobilenet(multiplier=1.0, classes=10)
elif model_name == 'mobilenet_v2':
    model = get_mobilenet_v2(multiplier=1.0, classes=10)
elif model_name == 'shufflenet':
    model = get_shufflenet(groups=2, classes=10)
else:
    assert 'illegal name'

# Loss and optimizer
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# # Save the model checkpoint
# torch.save(model.state_dict(), 'resnet.ckpt')
