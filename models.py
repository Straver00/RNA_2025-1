import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ----------------------
# Dataset Wrapper
# ----------------------
class TabularDataset(Dataset):
    """
    Dataset for tabular data: applies scaling and returns tensors.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: StandardScaler = None):
        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------
# Model Definitions
# ----------------------
class BaselinePerceptron(nn.Module):
    """
    Single-layer perceptron with configurable activation.
    Activation: 'sigmoid' for BCELoss or 'tanh' for MSELoss setups.
    """
    def __init__(self, input_dim: int, activation: str = 'sigmoid'):
        super(BaselinePerceptron, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("Unsupported activation: choose 'sigmoid' or 'tanh'")

    def forward(self, x):
        return self.act(self.linear(x))


class TwoLayerMLP(nn.Module):
    """
    MLP with two hidden layers, ReLU activations and dropout.
    """
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.3):
        super(TwoLayerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DeepBatchNormMLP(nn.Module):
    """
    Deeper MLP with BatchNorm and dropout after each layer.
    """
    def __init__(self, input_dim, layers=[128, 64, 32], dropout=0.2):
        super(DeepBatchNormMLP, self).__init__()
        modules = []
        prev_dim = input_dim
        for h in layers:
            modules += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h
        modules += [nn.Linear(prev_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

# ----------------------
# Training & Evaluation
# ----------------------
class Trainer:

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 criterion: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [],
                        'train_acc': [], 'val_acc': [], 'val_auc': []}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None,
            epochs: int = 20, threshold: float = 0.5):
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            train_losses, train_preds, train_targets = [], [], []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(y_batch.cpu().numpy())
            # Metrics
            y_pred_train = np.vstack(train_preds)
            y_true_train = np.vstack(train_targets)
            train_loss = np.mean(train_losses)
            train_acc = accuracy_score((y_pred_train > threshold), y_true_train)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses, val_preds, val_targets = [], [], []
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val = X_val.to(self.device)
                        y_val = y_val.to(self.device)
                        outputs = self.model(X_val)
                        loss = self.criterion(outputs, y_val)
                        val_losses.append(loss.item())
                        val_preds.append(outputs.cpu().numpy())
                        val_targets.append(y_val.cpu().numpy())
                y_pred_val = np.vstack(val_preds)
                y_true_val = np.vstack(val_targets)
                val_loss = np.mean(val_losses)
                val_acc = accuracy_score((y_pred_val > threshold), y_true_val)
                try:
                    val_auc = roc_auc_score(y_true_val, y_pred_val)
                except ValueError:
                    val_auc = float('nan')

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_auc'].append(val_auc)

                print(f"Epoch {epoch}/{epochs}"
                      f" - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}"
                      f" - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_auc: {val_auc:.4f}")
            else:
                print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
        return self.history