import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F

print("--- Script de Visualización Práctica 1 ---")

# =========================================================
# === 1. CONFIGURACIÓN DEL EXPERIMENTO (¡MODIFICA AQUÍ!) ===
# =========================================================

# --- Nombre del experimento a visualizar ---

EXPERIMENT_NAME = 'config3' 

# ---------------------------------------------------------

MODEL_PATH = f'{EXPERIMENT_NAME}_best_model.pth'
HISTORY_PATH = f'{EXPERIMENT_NAME}_history.npy'
BATCH_SIZE = 64 # Se recomienda usar el BATCH_SIZE más reciente

# Rutas del dataset (solo necesitamos 'test' para la matriz de confusión)
base_dir = os.getcwd()
test_dir = os.path.join(base_dir, 'dataset', 'seg_test', 'seg_test')

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# --- 2. DEFINICIÓN DEL MODELO (COPIADA DE train.py) ---
# Necesitamos la misma clase con la lógica de capas condicional (2 o 3)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, conv_layers):
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

        # Bloque Convolucional 3 (Debe coincidir con el modelo guardado)
        if self.conv_layers == 3:
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.drop3_conv = nn.Dropout(p=0.25)
            self.fc_input_size = 64 * 18 * 18
        else:
            self.fc_input_size = 32 * 37 * 37 

        # Capas Completamente Conectadas (Clasificador)
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        
        if self.conv_layers == 3:
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = self.drop3_conv(x)
        
        x = x.view(-1, self.fc_input_size)
        
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x
# --- Fin de la definición del modelo ---


# --- 3. CARGAR DATOS DE PRUEBA ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

try:
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
except Exception as e:
    print(f"Error cargando 'test_dataset': {e}")
    exit()

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(test_dataset.classes)
class_names = test_dataset.classes
print(f"Clases cargadas: {class_names}")


# --- 4. CARGAR MODELO ENTRENADO ---
# Determinar las capas usadas en el modelo guardado (asumimos 2 para las Configs 1, 2, 3)
# NOTA: Para Config. 4, DEBERÁS cambiar esta línea manualmente a CONV_LAYERS=3
CONV_LAYERS_LOAD = 2 

# Instanciamos la arquitectura y cargamos los pesos
model = SimpleCNN(num_classes=num_classes, conv_layers=CONV_LAYERS_LOAD).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except Exception as e:
    print(f"Error al cargar el modelo '{MODEL_PATH}': {e}")
    print(f"Asegúrate de que {EXPERIMENT_NAME} ha terminado su entrenamiento.")
    exit()
    
model.eval()


# --- 5. GENERAR GRÁFICAS DE PÉRDIDA Y PRECISIÓN ---
print(f"Cargando historial desde '{HISTORY_PATH}'...")
try:
    history = np.load(HISTORY_PATH, allow_pickle=True).item()
except Exception as e:
    print(f"Error al cargar el historial '{HISTORY_PATH}': {e}")
    exit()

print("Generando gráficas de Pérdida y Precisión...")
plt.figure(figsize=(12, 5))

# Gráfica de Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Precisión Entrenamiento')
plt.plot(history['test_acc'], label='Precisión Prueba (Validación)')
plt.title(f'Precisión - {EXPERIMENT_NAME}')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Pérdida Entrenamiento')
plt.plot(history['test_loss'], label='Pérdida Prueba (Validación)')
plt.title(f'Pérdida - {EXPERIMENT_NAME}')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig(f'{EXPERIMENT_NAME}_plots.png')
print(f"Gráficas guardadas en '{EXPERIMENT_NAME}_plots.png'")


# --- 6. GENERAR MATRIZ DE CONFUSIÓN ---
print("Generando Matriz de Confusión...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

print(f"\n--- Reporte de Clasificación ({EXPERIMENT_NAME}) ---")
report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
print(report)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Matriz de Confusión - {EXPERIMENT_NAME}')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Real')
plt.savefig(f'{EXPERIMENT_NAME}_confusion_matrix.png')
print(f"Matriz de confusión guardada en '{EXPERIMENT_NAME}_confusion_matrix.png'")

# Extraer precisión final
accuracy_report = classification_report(all_labels, all_preds, output_dict=True, target_names=class_names, zero_division=0)
overall_accuracy = accuracy_report['accuracy']
print(f"\nPRECISIÓN FINAL GLOBAL: {overall_accuracy:.4f}")

print("\n--- Fin del Script de Visualización ---")