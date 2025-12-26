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
# === CONFIGURACIÓN ACTUAL: CONFIGURACIÓN 5 (AUMENTO) ===
# =========================================================
EXPERIMENT_NAME = 'config5'  
CONV_LAYERS_LOAD = 3         
BATCH_SIZE_VIS = 64          
# =========================================================

FOLDER_PATH = EXPERIMENT_NAME
MODEL_PATH = os.path.join(FOLDER_PATH, f'{EXPERIMENT_NAME}_best_model.pth')
HISTORY_PATH = os.path.join(FOLDER_PATH, f'{EXPERIMENT_NAME}_history.npy')
PLOT_PATH = os.path.join(FOLDER_PATH, f'{EXPERIMENT_NAME}_plots.png')
CM_PATH = os.path.join(FOLDER_PATH, f'{EXPERIMENT_NAME}_confusion_matrix.png')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, conv_layers):
        super(SimpleCNN, self).__init__()
        self.conv_layers = conv_layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2); self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        if self.conv_layers == 3:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1); self.pool3 = nn.MaxPool2d(2, 2)
            self.fc_input_size = 64 * 18 * 18
        else: self.fc_input_size = 32 * 37 * 37 
        self.fc1 = nn.Linear(self.fc_input_size, 512); self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))); x = self.pool2(F.relu(self.conv2(x)))
        if self.conv_layers == 3: x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc_input_size); x = F.relu(self.fc1(x)); x = self.dropout(x)
        return self.fc2(x)

# Cargar Historial y graficar
history = np.load(HISTORY_PATH, allow_pickle=True).item()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['test_acc'], label='Test')
plt.title(f'Acc - {EXPERIMENT_NAME}'); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Train'); plt.plot(history['test_loss'], label='Test')
plt.title(f'Loss - {EXPERIMENT_NAME}'); plt.legend()
plt.tight_layout(); plt.savefig(PLOT_PATH)

# Evaluación
test_dir = os.path.join(os.getcwd(), 'dataset', 'seg_test', 'seg_test')
test_dataset = datasets.ImageFolder(test_dir, transform=transforms.Compose([
    transforms.Resize(150), transforms.CenterCrop(150), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_VIS, shuffle=False)
model = SimpleCNN(len(test_dataset.classes), CONV_LAYERS_LOAD).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        outputs = model(imgs.to(device)); _, p = torch.max(outputs, 1)
        all_preds.extend(p.cpu().numpy()); all_labels.extend(lbls.numpy())

print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
plt.figure(figsize=(10, 8)); sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.savefig(CM_PATH)
print("--- Visualización completa ---")