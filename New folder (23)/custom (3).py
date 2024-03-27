import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# Download the CIFAR-10 dataset to ./data
batch_size = 10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print("Downloading training data...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

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
PATH = 'cifar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

# Load your custom image
#custom_image_path = '/workspaces/AI/test_pics/car_test.png'  # Replace with the path to your image
# custom_image = Image.open(custom_image_path)

# Apply the necessary transformations
transform_custom = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to the expected input size
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),  # Discard the alpha channel if present
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your custom image
custom_image_path = '/workspaces/AI/test_pics/car_test.png'  # Replace with the path to your image
custom_image = Image.open(custom_image_path)
custom_image = transform_custom(custom_image)

# Add a batch dimension to the image
custom_image = custom_image.unsqueeze(0)

# Use the model to make predictions on the custom image
with torch.no_grad():
    model_output = net(custom_image)
    _, predicted_class = torch.max(model_output, 1)

# Display the custom image along with the predicted label
plt.imshow(np.transpose(custom_image.squeeze().numpy(), (1, 2, 0)))
predicted_class_name = classes[predicted_class.item()]
plt.title(f'Predicted Class: {predicted_class_name}')
print(f'Predicted Class: {predicted_class_name}')
plt.show()