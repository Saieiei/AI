import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Download the CIFAR-10 dataset to ./data
batch_size = 10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print("Downloading training data...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
print("Downloading testing data...")
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Our model will recognize these kinds of objects
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
PATH = './cifar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

# Function to predict class for a custom image
def predict_class(image_path):
    custom_image = Image.open(image_path)
    
    # Convert to RGB if the image has an alpha channel
    if custom_image.mode == 'RGBA':
        custom_image = custom_image.convert('RGB')

    transform_custom = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to the expected input size
        transforms.ToTensor()
    ])
    
    custom_image = transform_custom(custom_image)
    
    # Normalize manually since the custom image might have a different number of channels
    custom_image = (custom_image - 0.5) / 0.5
    custom_image = custom_image.unsqueeze(0)

    with torch.no_grad():
        model_output = net(custom_image)
        _, predicted_class = torch.max(model_output, 1)

    predicted_class_name = classes[predicted_class.item()]
    return predicted_class_name

# Test the model on a custom image
custom_image_path = '/workspaces/codespaces-jupyter/test_pics/car_test.jpg' 
predicted_class = predict_class(custom_image_path)
print(f'Predicted Class: {predicted_class}')
