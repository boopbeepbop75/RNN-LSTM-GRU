import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import Data_cleanup
import Dataset
import HyperParameters as H
import Model
import Utils as U
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

import pickle
import os

device = H.device
print(device)

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    X1 = torch.load(U.X1, weights_only=True)
    X2 = torch.load(U.X2, weights_only=True)
    y = torch.load(U.y, weights_only=True)

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    Data_cleanup.clean_data()
    X1 = torch.load(U.X1, weights_only=True)
    X2 = torch.load(U.X2, weights_only=True)
    y = torch.load(U.y, weights_only=True)

###Finish loading data###

'''print(X1.shape)
print(X2.shape)
print(y.shape)'''

def save_scalers(X1_scalers, X2_scalers, y_scaler, save_dir='./scalers/'):
    """
    Save all scalers to pickle files.
    
    Args:
        X1_scalers: List of scalers for X1 features
        X2_scalers: List of scalers for X2 features
        y_scaler: Scaler for target variable
        save_dir: Directory to save the scalers
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save X1 scalers
    for i, scaler in enumerate(X1_scalers):
        with open(f'{save_dir}X1_scaler_{i}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    # Save X2 scalers
    for i, scaler in enumerate(X2_scalers):
        with open(f'{save_dir}X2_scaler_{i}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    # Save y scaler
    with open(f'{save_dir}y_scaler.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)

Finance_Dataset = Dataset.Finance_Dataset(X1, X2, y)

sequences = Finance_Dataset._create_sequences()
#print(sequences)
X1 = sequences[0]
X2 = sequences[1]
y = sequences[2]

# Convert to NumPy
X1_np = X1.numpy() if isinstance(X1, torch.Tensor) else X1
X2_np = X2.numpy() if isinstance(X2, torch.Tensor) else X2
y_np = y.numpy() if isinstance(y, torch.Tensor) else y

# Split into training and testing sets (80% train, 20% test)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1_np, X2_np, y_np, test_size=0.2, shuffle=False  # Keep sequential order
)

# Scale each feature in X1 separately
X1_train_scaled = np.zeros_like(X1_train)
X1_test_scaled = np.zeros_like(X1_test)
X1_scalers = []

for i in range(X1_train.shape[-1]):  # Loop through each feature
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Create a new scaler
    X1_train_scaled[:, :, i] = scaler.fit_transform(X1_train[:, :, i])  # Fit & transform train
    X1_test_scaled[:, :, i] = scaler.transform(X1_test[:, :, i])  # Transform test using same scaler
    X1_scalers.append(scaler)

# Scale each feature in X2 separately
X2_train_scaled = np.zeros_like(X2_train)
X2_test_scaled = np.zeros_like(X2_test)
X2_scalers = []

for i in range(X2_train.shape[-1]):  # Loop through each discrete feature
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Create a new scaler
    X2_train_scaled[:, :, i] = scaler.fit_transform(X2_train[:, :, i])  # Fit & transform train
    X2_test_scaled[:, :, i] = scaler.transform(X2_test[:, :, i])  # Transform test using same scaler
    X2_scalers.append(scaler)

# Scale y separately
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

save_scalers(X1_scalers, X2_scalers, scaler_y)

# Convert back to tensors
X1_train_tensor = torch.tensor(X1_train_scaled, dtype=torch.float32)
X1_test_tensor = torch.tensor(X1_test_scaled, dtype=torch.float32)
X2_train_tensor = torch.tensor(X2_train_scaled, dtype=torch.float32)
X2_test_tensor = torch.tensor(X2_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

print(X1_train.shape)
print(X2_train.shape)
print(y_train.shape)

#Create Train / Test Datasets
train_dataset = Dataset.Finance_Sequence_Dataset(X1_train_tensor, X2_train_tensor, y_train_tensor)
test_dataset = Dataset.Finance_Sequence_Dataset(X1_test_tensor, X2_test_tensor, y_test_tensor)

print(train_dataset.__getitem__(0)[0].shape)

torch.save(test_dataset, U.model_predict)

train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

#Initialize model
model = Model.SimpleGRU(input_dim=9)
#model = Model.DebugRNN(input_dim=9)
model = model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

'''
Training Loop
#1. Forward Pass
#2. Calculate the loss on the model's predictions
#3. Optimizer
#4. Back Propagation using loss
#5. Optimizer step
'''

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(H.EPOCHS):
    print(f"Epoch: {epoch+1}\n---------")
    #Training
    training_loss = 0 #add up the loss over the course of a batch, then average it for the whole epoch
    #Add a loop to loop through the batched data
    #X: Image; Y: Label
    for batch, (X1, X2, y) in enumerate(train_loader):
        #Forward Pass
        X1 = X1.to(device)
        X2 = X2.to(device)
        y = y.to(device)
        y_preds = model(X1, X2)
        y_preds = y_preds.squeeze(1)

        #Loss calculation
        loss = loss_fn(y_preds, y)
        training_loss += loss.item()

        #Back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    training_loss /= len(train_loader)
    train_losses.append(training_loss)

    testing_loss, test_acc = 0, 0
    print("Testing the model...")
    model.eval()
    for batch, (X1, X2, y) in enumerate(test_loader):
        #Forward Pass
        X1 = X1.to(device)
        X2 = X2.to(device)
        y = y.to(device)
        y_preds = model(X1, X2)
        y_preds = y_preds.squeeze(1)

        #Loss calculation
        loss = loss_fn(y_preds, y)
        testing_loss += loss.item()

    testing_loss /= len(test_loader)
    test_acc /= len(test_loader)
    val_losses.append(testing_loss)

    #Evaluate model
    if testing_loss < best_val_loss:
        best_val_loss = testing_loss
        # Save the model's parameters (state_dict) to a file
        torch.save(model.state_dict(), (U.MODEL_FOLDER / (H.GRU_MODEL_NAME + '.pth')).resolve())
        with open((U.MODEL_FOLDER / (H.GRU_MODEL_NAME + '_loss.txt')).resolve(), 'w') as f:
            f.write(str(testing_loss))
        print(f'Saved best model with validation loss: {best_val_loss:.6f}')
        epochs_no_improve = 0  # Reset counter if improvement
    else:
        epochs_no_improve += 1
        print(f'Num epochs since improvement: {epochs_no_improve}')

        #stop training if overfitting starts to happen
        if epochs_no_improve >= H.PATIENCE:
            print("Early stopping")
            break

    print(f"Train loss: {training_loss:.6f} | Test loss: {testing_loss:.6f}")


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('LSTM Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

