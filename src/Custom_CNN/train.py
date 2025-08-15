import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from model import DrowsinessClassifier
import time
from pathlib import Path

def train_model(
    data_dir='../yolo/datasets/dataset_yolo',
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model, criterion, and optimizer
    model = DrowsinessClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training metrics
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, output_dir / 'latest_model.pth')
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s - "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_dir / 'confusion_matrix.png')
    
    # Print classification report
    class_names = ['Drowsy', 'Non-Drowsy']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model and visualizations saved in: {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Custom CNN for Drowsiness Detection')
    parser.add_argument('--data-dir', type=str, default='../yolo/datasets/dataset_yolo',
                      help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
