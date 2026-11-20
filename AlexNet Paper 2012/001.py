from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from sklearn.preprocessing import StandardScaler 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
iris = load_iris()
X = iris.data
y = iris.target
# 1. Data Loading & Pre-processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)

train_load = DataLoader(train_ds, batch_size=16, shuffle=True)
test_load = DataLoader(test_ds, batch_size=16, shuffle=False)

print("=== 5 mẫu đầu SAU StandardScaler ===")
print(X_train[:5])
print("Trung bình mỗi cột (~0):", X_train.mean(axis=0).round(4))
print("Độ lệch chuẩn mỗi cột (~1):", X_train.std(axis=0).round(4))
print("\n=== Dạng Tensor (N, C, L) ===")
print("X_train_tensor shape:", X_train_tensor.shape)
print("2 mẫu Tensor đầu:\n", X_train_tensor[:2])
xb, yb = next(iter(train_load))
print("\n=== 1 batch lấy từ DataLoader ===")
print("xb shape:", xb.shape, "| yb shape:", yb.shape)
print("Nhãn yb của batch này:", yb)

# 2. Build class TinyAlexNet rely on nn.Module

class TinyAlexNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2),
            nn.Conv1d(16, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(16, 3)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#Initualize Loss Function and Optimizer
model = TinyAlexNet()
out = model(xb)
print("Output shape:", out.shape) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Training Loop
for i in range(1, 601):
    med_Loss = 0
    model.train()
    for xb, yb in train_load:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward() 
        optimizer.step() 
        med_Loss += loss.item()
    med_Loss /= len(train_load)
    if i %20 == 0: print("Medium Loss: ", med_Loss)

#Evaluation
all_preds, all_targets = [], []
model.eval()
with torch.no_grad():
    for xb, yb in test_load:
        logits = model(xb)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_targets.extend(yb.tolist())
test_accuracy = accuracy_score(all_targets, all_preds)
print("Dự đoán (preds):", all_preds)
print("Đáp án (target):", all_targets)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")