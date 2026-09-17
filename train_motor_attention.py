import h5py
import numpy as np

with h5py.File('TestCaseManipulateBlockRandomized_full_10000.h5', 'r') as file:
    obs_group = file["obs"]
    asymmetric = obs_group["asymmetric"][:]        # (1000000, 25)
    goal = obs_group["goal"][:]                    # (1000000, 4)
    proprioception = obs_group["proprioception"][:]# (1000000, 48)
    touch = obs_group["touch"][:]                  # (1000000, 92)
    vision = obs_group["vision"][:]                # (1000000, 7)
###################################################################
X = np.concatenate(
    [asymmetric, goal, proprioception, touch, vision],
    axis=1
)

with h5py.File('TestCaseManipulateBlockRandomized_full_10000.h5', 'r') as file:
    y = file["action_distribution"][:]   
#########################################################################
X = np.concatenate([asymmetric, goal, proprioception, touch, vision], axis=1)

y = np.argmax(y, axis=2)  # (1000000, 20) class labels
print("y shape:", y.shape)
################
window_size = 16
X_sliding = np.array([X[i:i+window_size] for i in range(0, X.shape[0]-window_size+1)])
y_anto = y[window_size-1:]  

# action per timestep windowed 
y_prev = np.zeros_like(y); y_prev[1:] = y[:-1]
prev_sliding = np.array([y_prev[i:i+window_size] for i in range(0, y.shape[0]-window_size+1)])

X = X_sliding
y = y_anto
prev = prev_sliding
#########################################################################
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X = (X - X_mean) / X_std

# class labels 
y = y.astype(np.int64)
# previous action fed to the motor embedding
prev = prev.astype(np.float32)

#####################################################################
from sklearn.model_selection import train_test_split

#############################################
X_train, X_val, prev_train, prev_val, y_train, y_val = train_test_split(
    X, prev, y, test_size=0.1, random_state=42
)
####################################################
'''
# use a subset of trainig data
subset_size = 1000
X_train = X_train[:subset_size]
prev_train = prev_train[:subset_size]
y_train = y_train[:subset_size]

print(y_train)
'''
####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from motor_attention import MotorAttentionNet, N_ACTIONS, N_BINS

# Convert split data to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
prev_train = torch.tensor(prev_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)          # class labels

X_val = torch.tensor(X_val, dtype=torch.float32)
prev_val = torch.tensor(prev_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_ds = TensorDataset(X_train, prev_train, y_train)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

#  network 
net = MotorAttentionNet(embd=16, head_size=16, decay=0.95, use_motor=True)

criterion = nn.CrossEntropyLoss() # softmax over number of bins per joint
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)


def joint_ce(logits, targets):
    # cross-entropy over joints
    return criterion(logits.reshape(-1, N_BINS), targets.reshape(-1))

best_val = float("inf")
best_epoch = 0

with open("errors_motor_attention.txt", "w") as f:
    num_epochs = 200
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for batch_X, batch_prev, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_X, batch_prev)        # (B, 20, 11)
            loss = joint_ce(outputs, batch_y)         # softmax CE 
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        net.eval()
        with torch.no_grad():
            val_outputs = net(X_val, prev_val)                     # (Nval, 20, 11)
            val_loss = joint_ce(val_outputs, y_val).item()
            val_acc = (val_outputs.argmax(-1) == y_val).float().mean().item()

        # save the best model (by validation loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(net.state_dict(), "motor_attention_best.pt")

        # Epoch, Train Loss, Val Loss, Val Acc
        f.write(f"{epoch + 1}\t{epoch_loss:.6f}\t{val_loss:.6f}\t{val_acc:.6f}\n")
        f.flush()  # Forces write to disk so you can see it live
        print(f"Epoch {epoch+1}: train {epoch_loss:.4f} | "
              f"val {val_loss:.4f} | val_acc {val_acc:.4f}")

# final model 
torch.save(net.state_dict(), "motor_attention_final.pt")

# report best checkpoint
print(f"Best val loss {best_val:.4f} at epoch {best_epoch} -> motor_attention_best.pt")
print("saved motor_attention_final.pt")
