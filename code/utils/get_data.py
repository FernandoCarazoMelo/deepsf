import wandb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import namedtuple


def get_data(path, config):
  # Read the data from the file.

  with open(path + '/code_JS/folder_rawdata_processing/3-pipeline_files.pkl', 'rb') as fid:
    result = pickle.load(fid)

  # pacientes x gene_expression SFs
  TCGA_tpm_gn_RBPs = result['TCGA_tpm_gn_RBPs']
  # pacientes x gene_expression total genes
  TCGA_tpm_gn = result['TCGA_tpm_gn']
  # pacientes x isoform_expression
  TCGA_tpm_without_uniqueiso = result['TCGA_tpm_without_uniqueiso']
  # index x (Transcript_ID, Gene_ID, Transcrip_name, Gene_name, Biotype)
  getBM = result['getBM']

  # Transformación Log2 a la matriz de isoformas.
  TCGA_tpm_without_uniqueiso_log2p = np.log2(1+TCGA_tpm_without_uniqueiso)
  # getBM real.
  getBM = getBM.iloc[[
      a in TCGA_tpm_without_uniqueiso_log2p.columns for a in getBM.Transcript_ID], :]

  toy_genes = list(getBM['Gene_name'][:config.num_genes])  # + [ 'TP53']

  # getBM reducido.
  getBM = getBM.iloc[[a in toy_genes for a in getBM.Gene_name], :]
  getBM

  toy_Transcript_ID = list(getBM.Transcript_ID)
  TCGA_tpm_without_uniqueiso_log2p = TCGA_tpm_without_uniqueiso_log2p.loc[:,
                                                                          toy_Transcript_ID]

  # Creamos el input 2: pacientes x expresión de los genes de cada una de las isoformas (de la matriz de genes total).
  TCGA_tpm_gn_expr_each_iso = pd.DataFrame(np.zeros((TCGA_tpm_gn.shape[0], TCGA_tpm_without_uniqueiso_log2p.shape[1])),
                                           index=TCGA_tpm_gn.index, columns=list(getBM.Gene_name))

  for i in list(getBM.Gene_name):
    TCGA_tpm_gn_expr_each_iso[i] = TCGA_tpm_gn.loc[:, i]
  # pacientes x expresión de los genes de cada una de las isoformas.
  TCGA_tpm_gn_expr_each_iso.head()

  # Split in Training and Validation and Standarization of SFs expression
  df_train, df_validation = train_test_split(
      TCGA_tpm_gn_RBPs, test_size=config.test_size, random_state=0)

  # labels (we need the same patients so we use the same index selection)
  train_labels = TCGA_tpm_without_uniqueiso_log2p.loc[df_train.index]
  valid_labels = TCGA_tpm_without_uniqueiso_log2p.loc[df_validation.index]

  # gen_expr
  train_gn = TCGA_tpm_gn_expr_each_iso.loc[df_train.index]
  valid_gn = TCGA_tpm_gn_expr_each_iso.loc[df_validation.index]

  # Scale the SF input data:
  scaler_sfs = StandardScaler()  # Initialize
  # We put the content inside the scaler. For each feature mean and std.
  scaler_sfs.fit(df_train)

  scaledTrain_df = pd.DataFrame(scaler_sfs.transform(
      df_train), index=df_train.index, columns=df_train.columns)
  scaledValidation_df = pd.DataFrame(scaler_sfs.transform(
      df_validation), index=df_validation.index, columns=df_validation.columns)

  # Scale the gen_expr:
  scale_gn = StandardScaler()
  scale_gn.fit(train_gn)

  scaled_train_gn = pd.DataFrame(scale_gn.transform(
      train_gn), index=train_gn.index, columns=train_gn.columns)
  scaled_valid_gn = pd.DataFrame(scale_gn.transform(
      valid_gn), index=valid_gn.index, columns=valid_gn.columns)

  # Convert to PyTorch dataset
  train_ds = TensorDataset(torch.tensor(scaledTrain_df.values, dtype=torch.float32),
                           torch.tensor(train_labels.values,
                                        dtype=torch.float32),
                           torch.tensor(scaled_train_gn.values, dtype=torch.float32))

  val_ds = TensorDataset(torch.tensor(scaledValidation_df.values, dtype=torch.float32),
                         torch.tensor(valid_labels.values,
                                      dtype=torch.float32),
                         torch.tensor(scaled_valid_gn.values, dtype=torch.float32))

  train_loader = DataLoader(train_ds, config.batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, config.batch_size*2)
  
  DataPrep = namedtuple('DataPrep', ['train_ds', 'val_ds', 'train_loader', 'val_loader'])
  data_prep = DataPrep(train_ds, val_ds, train_loader, val_loader)

  return data_prep


