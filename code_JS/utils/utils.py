# Imports
from pickletools import optimize
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
import torch
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

# local modules
import modelsNN.modelsNN as modelsNN

def get_data(path, config):
    
    # Leer los datos (LUAD):
    result = pyreadr.read_r(path+'code_JS/data/input/LUAD/Parsing_luad.RData') #unlist Parsing_TCGA_LUAD:
    # result is a dictionary where keys are the name of objects.-> odict_keys(['TCGA_tpm', 'TCGA_tpm_gn', 'getBM', 'cond'])
    #Lista de los RBPs conocidos (SF genes).
    listRBPs = pd.read_excel(path+'code_JS/data/input/Table_S2_list_RBPs_eyras.xlsx', skiprows=2)
    

    #Estos son los datos de expresión de isoforma (TCGA_tpm) de genes (TCGA_tpm_gn). Además de la
    #información de tipo de tejido, normal vs tumoral (cond) y la información de la relación 
    #de genes e isoformas (getBM).

    TCGA_tpm = result["TCGA_tpm"] #expresión isoforma (transcrito) por paciente
    TCGA_tpm_gn = result["TCGA_tpm_gn"] #expresión gen por paciente (valor de tpm)
    getBM = result["getBM"] # matriz que te relaciona la isoforma (transcrito) con el gene_name
    cond = result["cond"] #para todos los pacientes (tejido) si son normal o tumoral.

    # Obtención de la matriz de genes de splicing (SF=RBPs) a partir de la matriz
    # de genes general:
    listRBPs_intersect = set(listRBPs['HGNC symbol']).intersection((set(TCGA_tpm_gn.index)))
    TCGA_tpm_gn_RBPs = TCGA_tpm_gn.loc[listRBPs_intersect] # matriz de genes SF por paciente (valor de tpm)

    # Eliminar isoformas únicas de la matriz de isoformas.
    num_iso_per_gn = getBM.groupby(['Gene_name']).count()['Transcript_ID'] # Número de isoformas por gen
    listgn_uniqueiso = num_iso_per_gn[num_iso_per_gn == 1].index.to_list() # Lista gn que solo tienen 1 isoforma:
    list_uniqueiso = getBM[getBM["Gene_name"].isin(listgn_uniqueiso)]["Transcript_ID"].to_list()
    listiso_intersect = set(TCGA_tpm.index.to_list()) - set(list_uniqueiso) # isoformas con las que nos vamos a quedar
    
    TCGA_tpm_without_uniqueiso = TCGA_tpm.loc[listiso_intersect]

    # Trasponemos las matriz para tener los datos de los pacientes en las filas
    TCGA_tpm_gn_RBPs = TCGA_tpm_gn_RBPs.T 
    TCGA_tpm_without_uniqueiso = TCGA_tpm_without_uniqueiso.T
    TCGA_tpm_gn = TCGA_tpm_gn.T
    
#     # Read the data from the file.
#     with open(path + '/code_JS/folder_rawdata_processing/3-pipeline_files.pkl', 'rb') as fid:
#         result = pickle.load(fid)

#     # pacientes x gene_expression SFs
#     TCGA_tpm_gn_RBPs = result['TCGA_tpm_gn_RBPs']
#     # pacientes x gene_expression total genes
#     TCGA_tpm_gn = result['TCGA_tpm_gn']
#     # pacientes x isoform_expression
#     TCGA_tpm_without_uniqueiso = result['TCGA_tpm_without_uniqueiso']
#     # index x (Transcript_ID, Gene_ID, Transcrip_name, Gene_name, Biotype)
#     getBM = result['getBM']

    # Transformación Log2 a la matriz de isoformas.
    TCGA_tpm_without_uniqueiso_log2p = np.log2(1+TCGA_tpm_without_uniqueiso)
    # getBM real.
    getBM = getBM.iloc[[a in TCGA_tpm_without_uniqueiso_log2p.columns for a in getBM.Transcript_ID], :]

    toy_genes = list(getBM['Gene_name'][:config.num_genes])  # + [ 'TP53']

    # getBM reducido.
    getBM = getBM.iloc[[a in toy_genes for a in getBM.Gene_name], :]

    toy_Transcript_ID = list(getBM.Transcript_ID)
    TCGA_tpm_without_uniqueiso_log2p = TCGA_tpm_without_uniqueiso_log2p.loc[:,toy_Transcript_ID]

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

    DataPrep = namedtuple('DataPrep', ['scaledTrain_df', 'train_labels', 'scaled_train_gn', 
                                       'scaledValidation_df', 'valid_labels', 'scaled_valid_gn', 
                                       'train_ds', 'val_ds', 'train_loader', 'val_loader', 'getBM'])
    
    data_prep = DataPrep(scaledTrain_df, train_labels, scaled_train_gn, scaledValidation_df, 
                         valid_labels, scaled_valid_gn, train_ds, val_ds, train_loader, val_loader, getBM)

    return data_prep

def build_optimizer(model, optimizer, learning_rate):
    if optimizer == 'sgd90':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == 'sgd70':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)
    elif optimizer == 'sgd50':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(), lr=learning_rate)
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


# def fit(epochs, lr, model, train_loader, val_loader, optimizer, weights=''):
#     history = []
#     #optimizer = opt_func(model.parameters(), lr)
#     for epoch in range(epochs):
#         # Training Phase
#         train_epoch_end = do_training(model, train_loader, optimizer)
#         # Validation phase
#         val_epoch_end = evaluate(model, val_loader)
#         model.epoch_end(epoch, train_epoch_end, val_epoch_end)
#         history.append(val_epoch_end)
#     return history


def fit(epochs, lr, model, train_loader, val_loader, optimizer, 
         hyperparameters, if_wandb):
    
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
        
   
#     if if_wandb:
#     # Save the model in the exchangeable ONNX format
#         torch.onnx.export(model, "model.onnx")
#         wandb.save("model.onnx")
        
    return history


def f_rmse_weighted(input, target, weights):
    return torch.sum(weights * (input - target) ** 2)/input.nelement()

# Plots
#Plot 1
def val_loss_vs_epoch(history, if_wandb): # Función que plotea el val_loss por epoch  
    losses = [r['val_loss'] for r in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    plt.title('val_loss vs. epochs')
    
    if if_wandb:
        wandb.log({"val_loss_vs_epoch": wandb.Image(plt)})

class ReturnPlotSolution(object):
    def __init__(self, cor_total, cor_trans):
        self.cor_total =  cor_total
        self.cor_trans = cor_trans  
    
#Plot2
def plot_pred_vs_real(x,y,g, model,if_wandb, data_type): # Función que te plotea la predicción del 
      #modelo versus el real y te calcula el coeficiente de correlación total y 
      #por transcrito.

    plt.figure(figsize=(15,10))

    Y_df = y.copy() # Matriz Real (para la correlación de transcritos)
  
    x = model(torch.Tensor(x.values), torch.Tensor(g.values)).detach().numpy()
    X_df = x.copy() # Matriz Pred (para la correlación de transcritos)
  
    x = x.flatten()
    y = y.values.flatten()

    fig = sns.regplot(x=x, y=y, scatter_kws = {'alpha':0.1})
    cor_total = stats.spearmanr(x,y)[0] # total correlation between real and pred

    # Correlación de transcritos:
    X_df = pd.DataFrame(X_df, columns = Y_df.columns, index = Y_df.index) # Matriz Pred

    cor_trans = []
    for i in Y_df.columns:
        cor_trans.append(stats.spearmanr(X_df.loc[:,i],Y_df.loc[:,i])[0])

    if if_wandb:
        wandb.log({"plot_pred_vs_real_{}".format(data_type): wandb.Image(plt), "cor_total_{}".format(data_type): cor_total,
                   "cor_trans_{}".format(data_type): cor_trans})
    return ReturnPlotSolution(cor_total, cor_trans)

#Boxplot 1
def corr_vs_biotype(getBM, train_labels, cor_values,if_wandb, data_type):
    
    plt.figure(figsize=(20,15))                   
    list_biotype = getBM[getBM.Transcript_ID.isin(train_labels.columns.values)]['Biotype'].values

    data = {'trans':train_labels.columns.values,
          'corr':  cor_values,
          'biotype': list_biotype}                     
                         

    plot_df = pd.DataFrame(data)
    g = sns.catplot(y="biotype", x="corr",
                  data=plot_df, palette="Set3",
                orient="h", height=7, aspect=3,
                kind="violin", dodge=True, cut=0, bw=.2)
                       
    if if_wandb:
        wandb.log({"corr_vs_biotype_{}".format(data_type): wandb.Image(plt)})
#         wandb.log({f"corr_vs_biotype_{data_type} sdfsdf": wandb.Image(plt)})
    
class ReturnPlotFinalSolution(object):
    def __init__(self, solution_train_cor_total, solution_train_cor_trans,
                                  solution_val_cor_total, solution_val_cor_trans):
        
        self.solution_train_cor_total = solution_train_cor_total
        self.solution_train_cor_trans = solution_train_cor_trans 
        self.solution_val_cor_total = solution_val_cor_total
        self.solution_val_cor_trans = solution_val_cor_trans 
    
def plot_results(history, scaledTrain_df, train_labels, scaled_train_gn,
                 scaledValidation_df, valid_labels, scaled_valid_gn, model, getBM,
                 if_wandb):

    val_loss_vs_epoch(history,if_wandb)
    solution_train = plot_pred_vs_real(scaledTrain_df, train_labels, scaled_train_gn,
                                       model,if_wandb, 'training') # training
    solution_val = plot_pred_vs_real(scaledValidation_df,valid_labels, scaled_valid_gn,
                                     model,if_wandb,'validation') # validation
    
    # https://stackoverflow.com/questions/59002624/why-i-get-nan-in-spearman-correlation-in-python
    corr_vs_biotype(getBM, train_labels, solution_train.cor_trans,if_wandb, 'training')
    corr_vs_biotype(getBM, valid_labels, solution_val.cor_trans,if_wandb, 'validation')
    
     
    return ReturnPlotFinalSolution(solution_train.cor_total, solution_train.cor_trans,
                                  solution_val.cor_total, solution_val.cor_trans)
                         
   