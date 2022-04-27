# Imports
from pickletools import optimize
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
#import torchvision
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
import pyreadr
import os
from captum.attr import DeepLift

# local modules
import modelsNN.modelsNN as modelsNN


def join_different_tumor_data(config, path, listRBPs):

    TCGA_tpm_gn_RBPs = pd.DataFrame()
    TCGA_tpm = pd.DataFrame()
    TCGA_tpm_gn = pd.DataFrame()
    cond = pd.DataFrame()

    for i in config.tumor_type:
        result = pyreadr.read_r(path+'/processed/'+f'Parsing_TCGA_{i}.RData')

        one_TCGA_tpm = result["TCGA_tpm"] #isoform expression per patient in tpm
        one_TCGA_tpm_gn = result["TCGA_tpm_gn"] #gen expression per patient in tpm
        
        one_cond = result["cond"] 
        one_cond['cond'] = i + '_' + one_cond['cond'].astype(str)#for all the patients if they are normal or tumor and to which tumor type they belong.

        # Obtaining the datafame of the expression of the splicing genes (SF = RBPs) from the general gene dataframe:
         
        listRBPs_intersect = set(listRBPs['HGNC symbol']).intersection((set(one_TCGA_tpm_gn.index)))
        one_TCGA_tpm_gn_RBPs = one_TCGA_tpm_gn.loc[listRBPs_intersect] # Dataframe of the expression of the SF genes.

        # Join the data of the different tumors:
        TCGA_tpm_gn_RBPs = pd.concat([TCGA_tpm_gn_RBPs, one_TCGA_tpm_gn_RBPs], axis = 1)
        TCGA_tpm = pd.concat([TCGA_tpm, one_TCGA_tpm], axis = 1)
        TCGA_tpm_gn = pd.concat([TCGA_tpm_gn, one_TCGA_tpm_gn], axis = 1)
        cond = pd.concat([cond, one_cond], axis = 0)

    getBM = result["getBM"] # Dataframe that relates the isoform with the gene and has more information.
 
     #""" Esta siempre es la misma para diferentes pacientes?? """
  
    DataRead = namedtuple('DataRead', ['TCGA_tpm_gn_RBPs', 'TCGA_tpm', 'TCGA_tpm_gn', 'cond', 'getBM'])
    data_read = DataRead(TCGA_tpm_gn_RBPs, TCGA_tpm, TCGA_tpm_gn, cond, getBM)

    return data_read


def get_data_AE(path,config):
    
    # List of splicing genes
    listRBPs = pd.read_excel(path+'/external/Table_S2_list_RBPs_eyras.xlsx', skiprows=2)
    
    # Read and join the data of different cancer types
    data_read = join_different_tumor_data(config, path, listRBPs)
    
    TCGA_tpm_gn = data_read.TCGA_tpm_gn.T
    cond = data_read.cond
    getBM = data_read.getBM

    getBM_prot_cod = getBM[getBM.Biotype == 'protein_coding']  # getBM with just protein coding genes
    
    gn_prot_cod_list = list(getBM_prot_cod.Gene_name.unique()) # Lista de los protein coding genes.
    
    # 2) Filtrar la matriz de genes de expresión con solo los protein coding:
    TCGA_tpm_gn_prot_cod = TCGA_tpm_gn.loc[:,gn_prot_cod_list]
    
    # 3) Split in training and validation and create Data Loader Tensor
    df_train, df_validation = train_test_split(TCGA_tpm_gn_prot_cod, test_size=0.2, random_state=0)

    # Convert to PyTorch dataset
    train_ds = TensorDataset(torch.tensor(df_train.values, dtype=torch.float32),
                             torch.tensor(df_train.values, dtype=torch.float32))

    val_ds = TensorDataset(torch.tensor(df_validation.values, dtype=torch.float32),
                           torch.tensor(df_validation.values, dtype=torch.float32))

    train_loaderAE = DataLoader(train_ds, config.batch_size, shuffle=True)
    val_loaderAE = DataLoader(val_ds, config.batch_size*2)
    
    DataAE = namedtuple('DataAE', ['getBM_prot_cod', 'TCGA_tpm_gn_prot_cod', 'train_loaderAE', 'val_loaderAE'])
    
    data_AE = DataAE(getBM_prot_cod, TCGA_tpm_gn_prot_cod, train_loaderAE, val_loaderAE)

    return data_AE
    
def get_data(path, config):
    
    # List of splicing genes
    listRBPs = pd.read_excel(path+'/external/Table_S2_list_RBPs_eyras.xlsx', skiprows=2)
    
    # Read and join the data of different cancer types
    data_read = join_different_tumor_data(config, path, listRBPs)
    
    TCGA_tpm_gn_RBPs = data_read.TCGA_tpm_gn_RBPs
    TCGA_tpm = data_read.TCGA_tpm
    TCGA_tpm_gn = data_read.TCGA_tpm_gn
    cond = data_read.cond
    getBM = data_read.getBM

    # Remove unique isoforms from the isoform dataframe
    num_iso_per_gn = getBM.groupby(['Gene_name']).count()['Transcript_ID'] # number of isoforms per gene
    listgn_uniqueiso = num_iso_per_gn[num_iso_per_gn == 1].index.to_list() # list of genes that have only one isoform.
    list_uniqueiso = getBM[getBM["Gene_name"].isin(listgn_uniqueiso)]["Transcript_ID"].to_list() # list of unique isoforms
    listiso_intersect = set(TCGA_tpm.index.to_list()) - set(list_uniqueiso) # list of transcripts with more than one isoform
    
    TCGA_tpm_without_uniqueiso = TCGA_tpm.loc[listiso_intersect] # Dataframe of the isoform expression without the transcripts with one isoform.

    # We transpose the dataframes to have the data of the patients in the axis 0 (rows):
    TCGA_tpm_gn_RBPs = TCGA_tpm_gn_RBPs.T 
    TCGA_tpm_without_uniqueiso = TCGA_tpm_without_uniqueiso.T
    TCGA_tpm_gn = TCGA_tpm_gn.T
    
    # Apply a Log2 transformation to the dataframe of isoform expression:
    TCGA_tpm_without_uniqueiso_log2p = np.log2(1+TCGA_tpm_without_uniqueiso)
    
    # Filter getBM. Only has the information of the transcripts that we have in the dataframe of the isoform expression:
    getBM = getBM.iloc[[a in TCGA_tpm_without_uniqueiso_log2p.columns for a in getBM.Transcript_ID], :]

    # Selecting the number of genes to use in the model:
    if config.if_toy:
        selected_genes = list(getBM['Gene_name'][:config.num_genes])

    else: # We only chose the genes related to the development of cancer
        listTumorGenes = pd.read_excel(path+'/external/Table_S6_S5_Cancer_gener_eyras.xlsx', skiprows=2)
        selected_genes = list(listTumorGenes['HGNC symbol'])
        config.num_genes = len(selected_genes) # fill the config.num_genes data

    # getBM reduced with just the information of the selected genes:
    getBM = getBM.iloc[[a in selected_genes for a in getBM.Gene_name], :] 

    selected_Transcript_ID = list(getBM.Transcript_ID)
    TCGA_tpm_without_uniqueiso_log2p = TCGA_tpm_without_uniqueiso_log2p.loc[:,selected_Transcript_ID] # Filter the dataframe of isoform expression by the selected transcripts.

    # Creation of the input2 with size equal to patients x expression of the genes of each of the isoforms:
    TCGA_tpm_gn_expr_each_iso = pd.DataFrame(np.zeros((TCGA_tpm_gn.shape[0], TCGA_tpm_without_uniqueiso_log2p.shape[1])),
                                           index=TCGA_tpm_gn.index, columns=list(getBM.Gene_name))

    for i in list(getBM.Gene_name):
        TCGA_tpm_gn_expr_each_iso[i] = TCGA_tpm_gn.loc[:, i]
    # patients x expression of the genes of each of the isoforms.
    

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

    DataPrep = namedtuple('DataPrep', ['scaledTrain_df', 'train_labels', 'scaled_train_gn', 
                                       'scaledValidation_df', 'valid_labels', 'scaled_valid_gn', 
                                       'train_ds', 'val_ds', 'train_loader', 'val_loader', 'getBM'])
    
    data_prep = DataPrep(scaledTrain_df, train_labels, scaled_train_gn, scaledValidation_df, 
                         valid_labels, scaled_valid_gn, train_ds, val_ds, train_loader, val_loader, getBM)

    return data_prep

def build_optimizer(model, optimizer, learning_rate):
    if optimizer == 'sgd90':
        # Stochastic gradient descent is extremely basic and is seldom used now. One problem is with the 
        # worldwide learning rate related to an equivalent . Hence it doesn’t work well when the parameters are 
        # in several scales since a coffee learning rate will make the training slow while an outsized learning 
        # rate might cause oscillations. Also, Stochastic gradient descent generally has a hard time escaping the
        # saddle points. Adagrad, Adadelta, RMSprop, and ADAM generally handle saddle points better. SGD with
        # momentum renders some speed to the optimization and also helps escape local minima better.
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    elif optimizer == 'sgd70':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)
    elif optimizer == 'sgd50':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    elif optimizer == 'asgd':
        # It Implements Averaged Stochastic Gradient Descent(ASGD) algorithm
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, lambd=0.0001, alpha=0.75)
    elif optimizer == 'adam':
        # adaptive Moment Estimation, it combines the good properties of Adadelta and 
        #RMSprop optimizer into one and hence tends to do better for most of the problems.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'adagrad':
        # Short for adaptive gradient, penalizes the learning rate for parameters that are frequently updated,
        # instead, it gives more learning rate to sparse parameters, parameters that are not updated as 
        # frequently.
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer == 'adamW':
        # an improved version of Adam class called AdamW in which weight decay is performed only after controlling the parameter-wise step size.
        #AdamW yields better training loss, that means the models generalize much better than models 
        # trained with Adam allowing the remake to compete with stochastic gradient descent with momentum.
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    return optimizer

def get_model(config, data_prep):
    inp, out, gn = next(iter(data_prep.train_loader))
    
    if config.modelNN == 'DeepSF':
        model = modelsNN.DeepSF(n_inputs=inp.shape[1], n_outputs=out.shape[1])
        
    elif config.modelNN == 'DeepSF_2hidden':
        model = modelsNN.DeepSF_2hidden(n_inputs=inp.shape[1], n_outputs=out.shape[1])
    return model


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def do_training(model, train_loader, optimizer):
    outputs = [model.training_step(batch, optimizer) for batch in train_loader]
    return model.training_epoch_end(outputs)


def fit(epochs, model, train_loader, val_loader, optimizer, 
         hyperparameters, if_wandb, path, model_name):
    
#     os.chdir(path)
#     os.chdir("..") # To go from /deepsf/data to /deepsf
#     path = os.getcwd()
    
    history = []
    if if_wandb:
        wandb.login()
        wandb.init(project="tutorial_joseba", config=hyperparameters)
        wandb.watch(model, criterion=F.mse_loss, log="all", log_freq=10)
        
    for epoch in range(epochs):
        # Training Phase
        train_epoch_end = do_training(model, train_loader, optimizer)
        # Validation phase
        val_epoch_end = evaluate(model, val_loader)
        model.epoch_end(epoch, train_epoch_end, val_epoch_end)
        history.append(val_epoch_end)
        
        if if_wandb:
            wandb.log({"loss": train_epoch_end, "loss_val": val_epoch_end, "epoch": epoch})
        
    if if_wandb:
        # Save the model
        #torch.save(model.state_dict(), path+'/code_JS/wandb/model.pth')
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_name))  
        #Load: model = MyModelDefinition(args)
        #      model.load_state_dict(torch.load('load/from/path/model.pth')) 
        
    else:
        torch.save(model.state_dict(), os.path.join(path, model_name)) 
        
    return history


def do_deeplift(model, X_test, gn_test, y_test, data_prep):
# igual lo del target no lo tenemos bien, mira aquí: https://github.com/pytorch/captum/issues/482

    base_x_test = np.median(X_test, axis = 0)
    base_x_test = torch.tensor(base_x_test).float()
    base_x_test = torch.reshape(base_x_test,(1,base_x_test.size()[0]))

    base_gn_test = np.median(gn_test, axis = 0)
    base_gn_test = torch.tensor(base_gn_test).float()
    base_gn_test = torch.reshape(base_gn_test,(1,base_gn_test.size()[0]))

    algorithm = DeepLift(model)
    df = pd.DataFrame(np.zeros((1,X_test.shape[1])), columns=X_test.columns.to_list()) # Output node x input features
    for i in range(0, y_test.shape[1]):

        attr_test = algorithm.attribute(
            inputs = (torch.tensor(X_test.values).float(),
            torch.tensor(gn_test.values).float()), 
            baselines = (base_x_test, base_gn_test),
            target=i)

        attr_test_sum = attr_test[0].detach().numpy().sum(0)

        if attr_test_sum.sum() == 0: #Para que no haya nulos.
            attr_test_norm_sum = attr_test_sum
        else:
            attr_test_norm_sum = attr_test_sum / np.linalg.norm(attr_test_sum, ord=1)

        df.loc[i,:] = attr_test_norm_sum

    df = df.T
    df.columns = data_prep.valid_labels.columns.to_list()
    return df

