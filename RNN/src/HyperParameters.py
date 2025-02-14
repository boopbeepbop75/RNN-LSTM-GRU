import torch

### Other Hyperparameters ###

### MODEL HYPER PARAMETERS ###
#Model name
LSTM_MODEL_NAME = 'LSTM_Model_0'
RNN_MODEL_NAME = 'RNN_Model_0'
GRU_MODEL_NAME = 'GRU_Model_0'
EPOCHS = 20
PATIENCE = 20

#Cuda
device = "cuda" if torch.cuda.is_available() else "cpu"