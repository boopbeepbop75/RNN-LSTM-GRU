import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import HyperParameters
from sklearn.metrics.pairwise import cosine_similarity

import Data_cleanup

def calculate_cosine_similarity(genres_movie_1, genres_movie_2):
    vector_1 = np.array(genres_movie_1)
    vector_2 = np.array(genres_movie_2)
    return cosine_similarity([vector_1], [vector_2])[0][0]



if __name__ == "__main__":
    Data_cleanup.clean_data()