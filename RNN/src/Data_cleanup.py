import pandas as pd
import numpy as np
import torch

import preprocessing_functions
import HyperParameters
import Utils as U
import random
import time
from sklearn.preprocessing import MinMaxScaler

def minmax_scale(data):
    """
    Normalize data to range [0, 1] using min-max scaling.
    
    Parameters:
    data (numpy.ndarray): Input array to normalize
    
    Returns:
    numpy.ndarray: Normalized array where values are scaled to [0, 1]
    """
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Handle the case where all values are the same
    if max_val == min_val:
        return np.zeros_like(data)
        
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

def load_data():
    df = pd.read_csv(U.data_raw)
    #print(df.isna().sum()) 
    return df

def preprocess_data(df):
    """'AAPL.Open', 'AAPL.High', 'AAPL.Low', 'AAPL.Close',
       'AAPL.Volume', 'AAPL.Adjusted', 'dn', 'mavg', 'up', 'direction', 'year',
       'month', 'day'"""
    direction_map = {'Increasing': 0, 'Decreasing': 1}
    df['direction'] = df['direction'].map(direction_map).astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df = df.drop(['Date'], axis=1)
    print(df.columns)
    data_x1 = [
        df['AAPL.Open'].values,
        df['AAPL.High'].values,
        df['AAPL.Low'].values,
        df['AAPL.Volume'].values,
        df['AAPL.Adjusted'].values,
        df['dn'].values,
        df['mavg'].values,
        df['up'].values
    ]
    data_x2 = [
        df['direction'].values,
        df['year'].values,
        df['month'].values,
        df['day'].values
    ]
    data_y = df['AAPL.Close'].values
    #Convert data to tensors based on their types 
    """
    X1 = qunatitative
    X2 = discrete
    y = labels (closing price)
    """
    X1 = torch.zeros(len(data_x1), len(data_x1[0]))
    X2 = torch.zeros(len(data_x2), len(data_x2[0]))
    y = torch.zeros(1, len(data_y))
    for col in range(len(data_x1)):
        X1[col, :] = torch.from_numpy(data_x1[col].astype(np.float32))
    for col in range(len(data_x2)):
        X2[col, :] = torch.from_numpy(data_x2[col].astype(np.float32))
    y = torch.from_numpy(data_y.astype(np.float32))
    X2 = X2.to(torch.long)
    print(X1.shape)
    print(X2.shape)
    print(y.shape)
    '''print(X1)
    print(X2)
    print(y)'''
    return X1, X2, y

def clean_data():
    df = load_data()
    X1, X2, y = preprocess_data(df)
    torch.save(X1, U.X1)
    torch.save(X2, U.X2)
    torch.save(y, U.y)

if __name__ == "__main__":
    clean_data()
