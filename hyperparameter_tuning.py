import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import numpy as np
import itertools

from train_loader import get_dataloaders

# Kleine DataLoader für schnelles Hyperparameter-Tuning
def get_small_dataloaders(batch_size, fraction=0.1):
    train_loader, val_loader, _ = get_dataloaders(batch_size)

    train_subset = Subset(
        train_loader.dataset,
        np.random.choice(len(train_loader.dataset),
                         int(fraction * len(train_loader.dataset)),
                         replace=False)
    )

    val_subset = Subset(
        val_loader.dataset,
        np.random.choice(len(val_loader.dataset),
                         int(fraction * len(val_loader.dataset)),
                         replace=False)
    )

    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    )


#Flexibles CNN für Filtervariation
class SimpleCNN(nn.Module):
    def __init__(self, c1=32, c2=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c2 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# --------------------------------------------------
# Trainings- und Evaluierungsfunktion
# --------------------------------------------------
def train_and_evaluate(config, num_epochs=10):
    train_loader, val_loader = get_small_dataloaders(
        batch_size=config["batch_size"],
        fraction=0.1  # 10 % der Daten
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(config["c1"], config["c2"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # Validation Accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# --------------------------------------------------
# Hyperparameterräume definieren
# --------------------------------------------------
c1_values = [32, 64]          # Filteranzahl Conv1
c2_values = [64, 128]         # Filteranzahl Conv2
lr_values = [0.001, 0.01]     # Lernraten
batch_values = [64, 128]      # Batchgrößen

# Alle Kombinationen
combinations = list(itertools.product(c1_values, c2_values, lr_values, batch_values))

# --------------------------------------------------
# Experimente durchführen
# --------------------------------------------------
results = []

for c1, c2, lr, batch_size in combinations:
    config = {
        "c1": c1,
        "c2": c2,
        "lr": lr,
        "batch_size": batch_size,
        "name": f"c1={c1},c2={c2},lr={lr},batch={batch_size}"
    }
    acc = train_and_evaluate(config, num_epochs=10)
    results.append({
        "Experiment": config["name"],
        "Filter": f"{c1}/{c2}",
        "Lernrate": lr,
        "Batchgröße": batch_size,
        "Val Accuracy (%)": round(acc, 2)
    })
    print(f"{config['name']}: {acc:.2f}%")

# --------------------------------------------------
# Ergebnisse als Tabelle
# --------------------------------------------------
df = pd.DataFrame(results)
print("\nErgebnisübersicht:")
print(df)

# Besten Hyperparameter markieren
best_idx = df["Val Accuracy (%)"].idxmax()
print("\nBestes Modell:")
print(df.loc[best_idx])

# --------------------------------------------------
# Plot der Ergebnisse mit Hervorhebung des besten Modells
# --------------------------------------------------
plt.figure(figsize=(15, 6))

colors = ['skyblue' if i != best_idx else 'orange' for i in range(len(df))]
bars = plt.bar(df["Experiment"], df["Val Accuracy (%)"], color=colors)

# Accuracy-Werte auf die Balken schreiben
for bar, acc in zip(bars, df["Val Accuracy (%)"]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8)

plt.ylabel("Validation Accuracy (%)")
plt.title("Hyperparameter-Kombinationen (10 Epochen, 10% Daten)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
