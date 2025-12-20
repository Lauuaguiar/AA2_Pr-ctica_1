import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

print("--- Script de Entrenamiento Práctica 1 ---")

# --- 1. CONFIGURACIÓN E HIPERPARÁMETROS ---

# -----------------------------------------------------
# CONFIGURACIÓN ACTUAL: CONFIGURACIÓN 3 (BATCH SIZE GRANDE)
# -----------------------------------------------------
EXPERIMENT_NAME = 'config3'
LEARNING_RATE = 0.0001  # Valor optimizado (manteniendo el mejor de la Config. 2)
EPOCHS = 50             # Máximo de épocas. Early Stopping decidirá.
BATCH_SIZE = 64         # ¡CAMBIO CLAVE! Minilote más grande
CONV_LAYERS = 2         # Número de capas convolucionales (2 ó 3)

# -----------------------------------------------------
# REGISTRO DE CONFIGURACIONES PREVIAS (Para referencia)
# -----------------------------------------------------

"""
# CONFIGURACIÓN 1 (BASE)
EXPERIMENT_NAME = 'config1'
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
CONV_LAYERS = 2

# CONFIGURACIÓN 2 (LR BAJO)
EXPERIMENT_NAME = 'config2'
LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32
CONV_LAYERS = 2
"""

# Rutas del dataset (Asegúrate que estas rutas son correctas para tu proyecto)
base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'dataset', 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'dataset', 'seg_test', 'seg_test')

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# --- 2. CARGA DE DATOS Y PREPROCESAMIENTO ---

# Valores de normalización
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transformaciones (Básicas, sin Data Augmentation por ahora)
data_transforms = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Creamos los Datasets usando ImageFolder
try:
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
except Exception as e:
    print(f"Error al cargar datos: {e}")
    exit()
    
# Creamos los DataLoaders (usando BATCH_SIZE de la configuración actual)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_classes = len(train_dataset.classes)
print(f"Clases detectadas ({num_classes}): {train_dataset.classes}")


# --- 3. DEFINICIÓN DEL MODELO (CNN) ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, conv_layers):
        """ conv_layers: 2 ó 3 """
        super(SimpleCNN, self).__init__()
        self.conv_layers = conv_layers
        
        # Bloque Convolucional 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.25)

        # Bloque Convolucional 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.25)

        # Bloque Convolucional 3 (Opcional, para Config. 4 en adelante)
        if self.conv_layers == 3:
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.drop3_conv = nn.Dropout(p=0.25)
            
            # Cálculo para 3 capas: 64 canales * 18 * 18 = 20736
            self.fc_input_size = 64 * 18 * 18
        else:
            # Cálculo para 2 capas: 32 canales * 37 * 37 = 43968
            self.fc_input_size = 32 * 37 * 37 

        # Capas Completamente Conectadas (Clasificador)
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.drop_fc = nn.Dropout(p=0.5) # Dropout para la capa FC
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Bloque 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Bloque 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Bloque 3 (Condicional)
        if self.conv_layers == 3:
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = self.drop3_conv(x)
        
        # Aplanado
        x = x.view(-1, self.fc_input_size)
        
        # Clasificador
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x

# --- 4. CLASE DE PARADA ANTICIPADA (Early Stopping) ---

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"   (Mejor modelo guardado en '{self.path}' con val_loss: {self.best_loss:.4f})")
        else:
            self.counter += 1
            print(f"   (EarlyStopping contador: {self.counter} de {self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True

# --- 5. FUNCIONES DE ENTRENAMIENTO Y PRUEBA (CORREGIDAS) ---
def train_step(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    total_samples = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        epoch_correct += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
    # CORRECCIÓN: total_samples es un entero, se elimina .item()
    return epoch_loss / total_samples, epoch_correct.double() / total_samples

def test_step(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            epoch_correct += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
    # CORRECCIÓN: total_samples es un entero, se elimina .item()
    return epoch_loss / total_samples, epoch_correct.double() / total_samples

# --- 6. BUCLE DE ENTRENAMIENTO PRINCIPAL ---

print("\n--- Iniciando el Experimento ---")

# Instanciar modelo (usando el número de capas de la configuración actual)
model = SimpleCNN(num_classes=num_classes, conv_layers=CONV_LAYERS).to(device)

# Definir Pérdida y Optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Instanciar Early Stopper con nombre de archivo dinámico
early_stopper = EarlyStopper(patience=5, path=f'{EXPERIMENT_NAME}_best_model.pth')

# Guardar historial para gráficas
history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

print(f"Entrenando {EXPERIMENT_NAME} (LR={LEARNING_RATE}, BS={BATCH_SIZE}, Capas={CONV_LAYERS})...")
for epoch in range(EPOCHS):
    
    train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_step(model, test_loader, criterion, device)
    
    # Guardar historial
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} --- "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Test Acc: {test_acc:.4f}")
    
    # Comprobación de Parada Anticipada
    early_stopper(test_loss, model)
    if early_stopper.early_stop:
        print("--- Parada anticipada activada ---")
        break

print("\n--- Entrenamiento Finalizado ---")

# Cargar el mejor modelo guardado (opcional, pero buena práctica)
model.load_state_dict(torch.load(early_stopper.path))
print(f"Mejor modelo cargado desde '{early_stopper.path}'")

# Guardar el historial con nombre dinámico
np.save(f'{EXPERIMENT_NAME}_history.npy', history)
print(f"Historial de entrenamiento guardado en '{EXPERIMENT_NAME}_history.npy'")

print("\n--- Fin del Script ---")