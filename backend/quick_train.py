# improved_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from app.model.cnn import DoodleNet
from app.utils.data_loader import QuickDrawDataset
import matplotlib.pyplot as plt


def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs, device):
    """
    Train the model with improved logging and validation
    """
    best_accuracy = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })

        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss/len(test_loader)
        val_acc = 100.*correct/total

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'models/doodle_model_best.pth')

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    return history


def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/training_history.png')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Categories to train on
    categories = [
        "apple", "banana", "cat", "dog", "elephant",
        "fish", "guitar", "house", "lion", "pencil",
        "pizza", "rabbit", "snake", "spider", "tree"
    ]

    # Create datasets with more samples
    train_dataset = QuickDrawDataset(
        root_dir='data/raw',
        categories=categories,
        train=True,
        download=True,
        sample_size=50000  # Increased sample size
    )

    test_dataset = QuickDrawDataset(
        root_dir='data/raw',
        categories=categories,
        train=False,
        download=False,
        sample_size=10000  # Increased sample size
    )

    # Create dataloaders with larger batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Increased batch size
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = DoodleNet(num_classes=len(categories))
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 50  # Increased epochs

    # Train model
    print("Starting training...")
    history = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # Save final model
    torch.save(model.state_dict(), 'models/doodle_model.pth')

    # Plot and save training history
    plot_training_history(history)

    print("Training completed!")
    print(f"Best model saved to: models/doodle_model_best.pth")
    print(f"Final model saved to: models/doodle_model.pth")
    print(f"Training history plot saved to: models/training_history.png")


if __name__ == '__main__':
    main()
