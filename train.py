import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

print("--- Script de Entrenamiento Práctica 1 ---")

# =========================================================
# === 1. REGISTRO DE CONFIGURACIONES (HISTORIAL) ========
# =========================================================

# --- CONFIGURACIÓN 1 (BASE) ---
# EXPERIMENT_NAME = 'config1' | LR = 0.001 | BS = 32 | Layers = 2

# --- CONFIGURACIÓN 2 (LR BAJO) ---
# EXPERIMENT_NAME = 'config2' | LR = 0.0001 | BS = 32 | Layers = 2

# --- CONFIGURACIÓN 3 (BATCH SIZE GRANDE) ---
# EXPERIMENT_NAME = 'config3' | LR = 0.0001 | BS = 64 | Layers = 2

# --- CONFIGURACIÓN 4 (ARQUITECTURA) ---
# EXPERIMENT_NAME = 'config4' | LR = 0.0001 | BS = 64 | Layers = 3

# ---------------------------------------------------------
# === CONFIGURACIÓN ACTUAL: CONFIGURACIÓN 5 (AUMENTO DE DATOS) ===
# ---------------------------------------------------------
EXPERIMENT_NAME = 'config5'  
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
CONV_LAYERS = 3              # Mantenemos la arquitectura de 3 capas
EPOCHS = 50                  
# =========================================================

# Gestión de carpetas
if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)
    print(f"Carpeta '{EXPERIMENT_NAME}' creada.")

MODEL_SAVE_PATH = os.path.join(EXPERIMENT_NAME, f'{EXPERIMENT_NAME}_best_model.pth')
HISTORY_SAVE_PATH = os.path.join(EXPERIMENT_NAME, f'{EXPERIMENT_NAME}_history.npy')

# Rutas del dataset
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'dataset', 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'dataset', 'seg_test', 'seg_test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Ejecutando: {EXPERIMENT_NAME} | Dispositivo: {device}")

# --- 2. CARGA DE DATOS CON DATA AUGMENTATION ---

# Transformaciones con aumento para ENTRENAMIENTO
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)), # Zoom aleatorio
    transforms.RandomHorizontalFlip(),                  # Volteo espejo
    transforms.RandomRotation(15),                      # Rotación ligera
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transformaciones estándar para PRUEBA
test_transforms = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
except Exception as e:
    print(f"Error cargando datos: {e}"); exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODELO CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, conv_layers):
        super(SimpleCNN, self).__init__()
        self.conv_layers = conv_layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        if self.conv_layers == 3:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.fc_input_size = 64 * 18 * 18
        else:
            self.fc_input_size = 32 * 37 * 37 

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        if self.conv_layers == 3:
            x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- 4. EARLY STOPPING ---
class EarlyStopper:
    def __init__(self, patience=5, path='model.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"   (Mejor modelo guardado en {self.path})")
        else:
            self.counter += 1
            print(f"   (EarlyStopping: {self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True

# --- 5. FUNCIONES DE PASO ---
def train_step(model, loader, criterion, optimizer):
    model.train()
    loss_acc, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_acc += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
    return loss_acc / total, correct.double() / total

def test_step(model, loader, criterion):
    model.eval()
    loss_acc, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_acc += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
    return loss_acc / total, correct.double() / total

# --- 6. EJECUCIÓN ---
model = SimpleCNN(len(train_dataset.classes), CONV_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopper(patience=5, path=MODEL_SAVE_PATH)

history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_step(model, train_loader, criterion, optimizer)
    te_loss, te_acc = test_step(model, test_loader, criterion)
    history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
    history['test_loss'].append(te_loss); history['test_acc'].append(te_acc)
    print(f"Epoch {epoch+1:02d} - Train Loss: {tr_loss:.4f}, Test Acc: {te_acc:.4f}")
    early_stopper(te_loss, model)
    if early_stopper.early_stop: break

np.save(HISTORY_SAVE_PATH, history)
print(f"--- Fin {EXPERIMENT_NAME} ---")