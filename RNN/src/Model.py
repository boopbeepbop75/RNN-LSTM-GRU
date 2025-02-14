import torch
import torch.nn.functional as F
import torch.nn as nn
import HyperParameters as H

device = H.device

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, dropout=.2):
        """
        A model that stacks RNN, LSTM, and GRU in sequence.

        Args:
        - input_dim (int): Number of input features per time step.
        - hidden_dim (int): Number of hidden units per layer.
        - output_dim (int): Number of output features.
        - num_layers (int): Number of layers for each RNN/LSTM/GRU.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Use LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Add batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Add dropout before the final layer
        self.dropout = nn.Dropout(dropout)
        
        # Multiple fully connected layers with ReLU
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # Keep only relevant features from x2
        drop_indices = [1, 2, 3]
        keep_indices = [i for i in range(x2.shape[-1]) if i not in drop_indices]
        x2 = x2[:, :, keep_indices]
        
        # Concatenate inputs
        x = torch.cat((x1, x2), dim=2)
        x = x.to(torch.float32)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        out = lstm_out[:, -1, :]
        
        # Apply batch normalization
        out = self.batch_norm(out)
        
        # Apply dropout and dense layers
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, dropout=.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Use GRU with dropout between layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Add batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Add dropout before the final layer
        self.dropout = nn.Dropout(dropout)
        
        # Multiple fully connected layers with ReLU
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # Keep only relevant features from x2
        drop_indices = [1, 2, 3]
        keep_indices = [i for i in range(x2.shape[-1]) if i not in drop_indices]
        x2 = x2[:, :, keep_indices]
        
        # Concatenate inputs
        x = torch.cat((x1, x2), dim=2)
        x = x.to(torch.float32)
        
        # Pass through GRU
        gru_out, _ = self.gru(x)
        
        # Take the last time step output
        out = gru_out[:, -1, :]
        
        # Apply batch normalization
        out = self.batch_norm(out)
        
        # Apply dropout and dense layers
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, dropout=.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Use RNN with dropout between layers
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Add batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Add dropout before the final layer
        self.dropout = nn.Dropout(dropout)
        
        # Multiple fully connected layers with ReLU
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # Keep only relevant features from x2
        drop_indices = [1, 2, 3]
        keep_indices = [i for i in range(x2.shape[-1]) if i not in drop_indices]
        x2 = x2[:, :, keep_indices]
        
        # Concatenate inputs
        x = torch.cat((x1, x2), dim=2)
        x = x.to(torch.float32)
        
        # Pass through RNN
        rnn_out, _ = self.rnn(x)
        
        # Take the last time step output
        out = rnn_out[:, -1, :]
        
        # Apply batch normalization
        out = self.batch_norm(out)
        
        # Apply dropout and dense layers
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out