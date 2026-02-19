import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIG =================
DATA_DIR = "data/"
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.0005
VAL_SPLIT = 0.2
PATIENCE = 5
USE_TRANSFER = True
# ==========================================

device = torch.device("cpu")

# ============ TRANSFORMS ============
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ============ MODEL ============
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


if USE_TRANSFER:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
else:
    model = CustomCNN(num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# ============ TRAINING ============
train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):

    # ----- TRAIN -----
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    train_loss /= len(train_loader)

    # ----- VALIDATION -----
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = correct / total
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_accuracy)
    val_accs.append(val_accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {train_accuracy:.4f} | "
          f"Val Acc: {val_accuracy:.4f} | "
          f"Val Loss: {val_loss:.4f}")

# ----- EARLY STOPPING & SAVING -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        # --- التعديل هنا ---
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_names': class_names, 
            'num_classes': num_classes
        }
        torch.save(checkpoint, "best_model.pth")
        print(f"✅ Saved best model with classes: {class_names}")
        # ------------------
        
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break


# ============ METRICS ============
cm = confusion_matrix(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

print("\nConfusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)

# ============ PLOTS ============
plt.figure()
plt.plot(train_losses)
plt.plot(val_losses)
plt.title("Loss")
plt.legend(["Train", "Validation"])
plt.show()

plt.figure()
plt.plot(train_accs)
plt.plot(val_accs)
plt.title("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()
