import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from train_loader import get_dataloaders


#CNN-Modell
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    # Device, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    # Listen für Plots
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            #Zurücksetzen des Gradienten
            optimizer.zero_grad()
            #Vorhersagen des Modells
            outputs = model(images)
            #Berechnet wie weit die Vorhersage von den Labels entfernt ist
            loss = criterion(outputs, labels)
            #Berechnet Gradienten der Parameter
            loss.backward()
            #Passt die Gewichtung an
            optimizer.step()
            
            #Statistik pro Batch
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        #Durchschnitt jedes Batches
        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        #Dieses Mal keine Gradientenberechnung
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                #Bekommt Output
                outputs = model(images)
                #Analyisiert Loss
                loss = criterion(outputs, labels)
                #Sammelt Statistik
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    print("\nTraining abgeschlossen.")
    torch.save(model.state_dict(), "cnn_cifar10_model.pth")

    # ==============================
    # Plots erstellen
    # ==============================
    epochs = range(1, num_epochs+1)

    # Loss-Plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss über Epochen")
    plt.legend()

    # Accuracy-Plot
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy über Epochen")
    plt.legend()

    plt.tight_layout()
    plt.show()
