import torch
import torch.nn as nn
import torch.nn.functional as F

class DrowsinessClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DrowsinessClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Fourth conv block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

    def predict(self, x):
        """Helper method for inference"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        return predicted_class, confidence
