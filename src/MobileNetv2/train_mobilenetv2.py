import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchvision.models import MobileNet_V2_Weights

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '../yolo/datasets/dataset_yolo')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_CLASSES = 2

# Data transforms
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.cuda() if torch.cuda.is_available() else model

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and validation
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda() if torch.cuda.is_available() else (inputs, labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.item())

    # Validation
    model.eval()
    val_running_loss, val_running_corrects = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda() if torch.cuda.is_available() else (inputs, labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = val_running_loss / len(val_dataset)
    val_acc = val_running_corrects.double() / len(val_dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc.item())

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mobilenetv2.pth'))

# Save final model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_mobilenetv2.pth'))

# Plot losses
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))

# Plot accuracies
plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_curve.png'))

# Confusion matrix and classification report
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

# Classification report
report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

print('Training complete. Metrics and plots saved to', OUTPUT_DIR) 