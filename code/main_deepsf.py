# %%
from sklearn import utils
import wandb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

# local functions
from utils.config import Config
import utils.utils as utils
from utils.get_data import get_data
from modelsNN.f_class_models_pytorch import evaluate, do_training, fit, DeepSF, DeepSF_2hidden

# %%

PATH_DEEPSF = '/content/gdrive/MyDrive/deepsf/code_JS/folder_rawdata_processing/'
PATH_DEEPSF = '..'

config = Config()

# %%
data_prep = get_data(PATH_DEEPSF, config)

# %%
