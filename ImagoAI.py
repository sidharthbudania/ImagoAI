import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

file_path = "/TASK-ML-INTERN.csv"
df = pd.read_csv(file_path)

df.drop(columns=['hsi_id'], inplace=True)

X = df.drop(columns=['vomitoxin_ppb']).values
y = df['vomitoxin_ppb'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

// improves MLP increase number of layers nodes increase complexity

class ImprovedMLP(nn.Module):
    def __init__(self, input_size):
        super(ImprovedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        self.feature_layer = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x, extract_features=False):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        features = self.feature_layer(x)
        if extract_features:
            return features
        return self.fc4(features)

input_size = X_pca.shape[1]
model = ImprovedMLP(input_size)

criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
X_train_features = model(X_train_tensor, extract_features=True).detach().numpy()
X_test_features = model(X_test_tensor, extract_features=True).detach().numpy()

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.5
)

xgb_model.fit(X_train_features, y_train.ravel())

y_pred_test = xgb_model.predict(X_test_features)

y_pred_test = target_scaler.inverse_transform(y_pred_test.reshape(-1, 1))
y_test_orig = target_scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_orig, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))
r2 = r2_score(y_test_orig, y_pred_test)

print(f"mae={mae:.4f}, Rmse={rmse:.4f}, r²={r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test_orig, y_pred_test, alpha=0.6, color='blue')
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], '--r', linewidth=2)
plt.xlabel("actual DON concentration")
plt.ylabel("predicted DON concentration")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.mean(X, axis=0), color='blue', linewidth=2)
plt.xlabel("spectral band")
plt.ylabel("average reflectance")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(X[:50, :], cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
plt.xlabel("spectral band index")
plt.ylabel("sample index")
plt.show()
