import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- IMPORTANTE: Reutilizar la definición del Modelo ---
# Necesitamos la *definición* de la clase SimpleCNN para poder cargar
# el modelo guardado. Debe ser IDÉNTICA a la de train.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.25)
        self.fc_input_size = 32 * 37 * 37 
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x
# --- Fin de la definición del modelo ---


print("--- Script de Visualización Práctica 1 ---")

# --- 1. CONFIGURACIÓN ---
MODEL_PATH = 'config1_best_model.pth'
HISTORY_PATH = 'config1_history.npy'
BATCH_SIZE = 32 # Usar el mismo batch size es buena idea

# Rutas del dataset (solo necesitamos 'test' para la matriz de confusión)
base_dir = os.getcwd()
test_dir = os.path.join(base_dir, 'dataset', 'seg_test', 'seg_test')

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# --- 2. CARGAR DATOS DE PRUEBA ---
# [Requisito del enunciado: Matriz de confusión sobre conjunto de prueba]

# Transformaciones (las mismas que en la prueba de train.py)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Cargar el dataset de prueba
try:
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
except Exception as e:
    print(f"Error cargando 'test_dataset': {e}")
    print("Asegúrate de que la ruta es correcta.")
    exit()

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(test_dataset.classes)
class_names = test_dataset.classes
print(f"Clases cargadas: {class_names}")


# --- 3. CARGAR MODELO ENTRENADO ---
# Cargamos el mejor modelo guardado por EarlyStopping

# Instanciamos la arquitectura
model = SimpleCNN(num_classes=num_classes).to(device)

# Cargamos los pesos (el estado)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except Exception as e:
    print(f"Error al cargar el modelo '{MODEL_PATH}': {e}")
    print("Asegúrate de que 'train.py' haya terminado y creado el archivo.")
    exit()
    
model.eval() # ¡Muy importante! Poner el modelo en modo evaluación


# --- 4. GENERAR GRÁFICAS DE PÉRDIDA Y PRECISIÓN ---
# [Requisito del enunciado, fuente: 36]
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
plt.title('Gráfica de Precisión (Accuracy)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Pérdida Entrenamiento')
plt.plot(history['test_loss'], label='Pérdida Prueba (Validación)')
plt.title('Gráfica de Pérdida (Loss)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig('config1_plots.png') # Guardamos las gráficas en un archivo
print("Gráficas guardadas en 'config1_plots.png'")
# plt.show() # Descomenta esto si quieres que se muestren en pantalla


# --- 5. GENERAR MATRIZ DE CONFUSIÓN ---
# [Requisito del enunciado, fuente: 37]
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

# Calcular la matriz de confusión
cm = confusion_matrix(all_labels, all_preds)

# Mostrar el reporte de clasificación (precisión, recall, f1-score)
print("\n--- Reporte de Clasificación (Conjunto de Prueba) ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Visualizar la matriz de confusión con Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión (Conjunto de Prueba)')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Real')
plt.savefig('config1_confusion_matrix.png')
print("Matriz de confusión guardada en 'config1_confusion_matrix.png'")
# plt.show() # Descomenta si quieres mostrarla

print("\n--- Fin del Script de Visualización ---")