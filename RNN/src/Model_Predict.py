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

def load_scalers(n_X1_features, n_X2_features, load_dir='./scalers/'):
    """
    Load all scalers from pickle files.
    
    Args:
        n_X1_features: Number of features in X1
        n_X2_features: Number of features in X2
        load_dir: Directory where scalers are saved
    
    Returns:
        X1_scalers: List of scalers for X1 features
        X2_scalers: List of scalers for X2 features
        y_scaler: Scaler for target variable
    """
    X1_scalers = []
    X2_scalers = []
    
    # Load X1 scalers
    for i in range(n_X1_features):
        with open(f'{load_dir}X1_scaler_{i}.pkl', 'rb') as f:
            X1_scalers.append(pickle.load(f))
    
    # Load X2 scalers
    for i in range(n_X2_features):
        with open(f'{load_dir}X2_scaler_{i}.pkl', 'rb') as f:
            X2_scalers.append(pickle.load(f))
    
    # Load y scaler
    with open(f'{load_dir}y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
    
    return X1_scalers, X2_scalers, y_scaler

X1_scalers, X2_scalers, y_scaler = load_scalers()

