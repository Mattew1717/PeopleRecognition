import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np
from pathlib import Path

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
image_path = Path("people/")
train_dir = image_path / "train"
test_dir = image_path / "test"

# Data transformations for training
data_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),  # Resize for crop margin
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random crop 80-100% of image
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomRotation(degrees=15),  # Random rotation ±15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.RandomGrayscale(p=0.1),  # Random grayscale with 10% probability
    transforms.RandomAffine(30, shear=10),  # Random affine transformations
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
])

# Basic transformation (for testing and inference)
basic_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])

# Load datasets
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

class_names = train_data.classes  # List of class names
class_dict = train_data.class_to_idx  # Dictionary of class indices

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

# DataLoader setup
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Load pre-trained ResNet152 model
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)

# Freeze all layers (optional)
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer to match the number of classes in the dataset
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# Move the model to the appropriate device
model = model.to(device)

# Training step function
def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# Testing step function
def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Main training function
def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 5):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

# Function to display image
def imshow(img):
    img = img.cpu()  # Move tensor to CPU
    img = img / 2 + 0.5  # Reverse normalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Change channel order for display
    plt.show()

# Function to process a single image for prediction
def processImage(image_path: str, transform=None):
    """Processes an image for model input."""
    image = Image.open(image_path)
    if transform:
        image_tensor = transform(image)
    return image_tensor

# Function to predict and plot the result for a given image
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names=None,
                        transform=None,
                        device=device):
    """Makes a prediction on an image and plots it."""
    
    # Process image
    image_tensor = processImage(image_path=image_path, transform=data_transform)
    real_image = processImage(image_path=image_path, transform=basic_transform)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    real_image = real_image.unsqueeze(0)

    # Move image tensor to device
    image_tensor = image_tensor.to(device)
    real_image = real_image.to(device)

    # Set model to eval mode
    model.eval()
    
    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1)
        pred_class = class_names[pred_label.item()]
        pred_prob = probs.max().item()

    # Plot the image with prediction
    plt.imshow(real_image.squeeze().permute(1, 2, 0).cpu())  # Remove batch dimension and adjust channels
    title = f"Pred: {pred_class} | Prob: {pred_prob:.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()

# Set loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
if __name__ == '__main__':
    NUM_EPOCHS = 5
    random.seed(42)

    model_results = train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS)
    # Optionally save the model
    # torch.save(model.state_dict(), 'resNet152_v6.pth')
