import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import os

## CNN

class SimpleCNN(nn.Module):
    name = "original"

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # The FER-2013 images are grayscale, so in_channels=1.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FER-2013 images are 48x48 pixels, so after two pooling layers, the size will be 12x12.
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes to classify
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model_original = SimpleCNN()
print(model_original)


# Variant 1: Add a third convolutional layer
class Variant1(SimpleCNN):
    name = "variant1"

    def __init__(self):
        super(Variant1, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # 3x3 kernel
        self.fc1 = nn.Linear(128 * 6 * 6, 128)  # Adjust the input size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Add a forward pass through the third convolutional layer
        x = x.view(-1, 128 * 6 * 6)  # Adjust the size
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model_variant1 = Variant1()
print(model_variant1)


# Variant 2: Change kernel sizes
class Variant2(SimpleCNN):
    name = "variant2"

    def __init__(self):
        super(Variant2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)  # 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1)  # 2x2 kernel
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Adjust the input size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Adjust the size
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model_variant2 = Variant2()
print(model_variant2)

## Preparing the dataset

# transforms
transform = transforms.Compose([
    transforms.Grayscale(),  # If the images are not already in grayscale
    transforms.Resize((48, 48)),  # Resize if the images are not already 48x48
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize with mean and std dev
])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(os.getcwd() + "/datasets-rev-1/final_clean/train", transform=transform)
test_dataset = datasets.ImageFolder(os.getcwd() + "/datasets-rev-1/final_clean/test", transform=transform)

# DataLoader
trainset_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
testset_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


## Training the Model

def train_model(train_loader, test_loader, model):
    # the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # Minimum number of epochs to train
    best_val_loss = float('inf')  # Initialize best validation loss for early stopping
    patience = 3  # Patience for early stopping
    patience_counter = 0  # Counter for early stopping

    print('Training model: ' + model.name)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Clear existing gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            running_loss += loss.item() * images.size(0)  # Accumulate loss

        # Calculate training loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # No gradient computation for validation
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # validation loss and accuracy
        val_loss /= len(test_loader.dataset)
        val_accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            print("Validation loss decreased. Saving the model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.getcwd() + '/saved models/best_model_' + model.name + '_retrained.pth')  # Save the best model
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            print(f"Patience counter: {patience_counter}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break  # If patience exceeded, stop training

    # After the loop, check if early stopping was triggered and if we didn't complete the minimum epochs
    if epoch + 1 < num_epochs:
        print(
            f"Early stopping triggered, but we didn't reach the minimum of {num_epochs} epochs. Continuing training...")
        # If we haven't completed the minimum epochs, we continue training until we do
        for epoch in range(epoch + 1, num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            # Training loss for the additional epochs
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

            model.eval()  # Validation phase in additional epochs
            with torch.no_grad():
                val_loss = 0.0
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(test_loader.dataset)
            print(f'Additional Validation Loss: {val_loss:.4f}')

    # Load the best model after training is complete
    model.load_state_dict(torch.load(os.getcwd() + '/saved models/best_model_' + model.name + '_retrained.pth'))
    model.eval()
    print("Training complete. Best model loaded.")


train_model(train_loader=trainset_loader, test_loader=testset_loader, model=model_original)
train_model(train_loader=trainset_loader, test_loader=testset_loader, model=model_variant1)
train_model(train_loader=trainset_loader, test_loader=testset_loader, model=model_variant2)
