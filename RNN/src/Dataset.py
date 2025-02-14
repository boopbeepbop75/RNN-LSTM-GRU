import torch
from torch.utils.data import Dataset

class Finance_Dataset(Dataset):
    def __init__(self, X1, X2, y, seq_length=30):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X1)
    
    def __getitem__(self, idx):
        return self.X1[:, idx], self.X2[:, idx], self.y[idx]
    
    def _create_sequences(self):
        """Create sequences for X1, X2, and y."""
        X1_seqs, X2_seqs, y_seqs = [], [], []
        
        for i in range(self.X1.shape[1] - self.seq_length):  # Iterate over observations
            X1_seqs.append(self.X1[:, i:i + self.seq_length].T)  # (seq_length, features)
            X2_seqs.append(self.X2[:, i:i + self.seq_length].T)  # (seq_length, features)
            y_seqs.append(self.y[i + self.seq_length])  # Next value prediction
        
        return torch.stack(X1_seqs), torch.stack(X2_seqs), torch.stack(y_seqs)
    

class Finance_Sequence_Dataset(Dataset):
    def __init__(self, X1, X2, y): 
        self.X1 = X1
        self.X2 = X2
        self.y = y
    
    def __len__(self):
        return len(self.X1)
    
    def __getitem__(self, idx):
        return self.X1[idx, :], self.X2[idx, :], self.y[idx]