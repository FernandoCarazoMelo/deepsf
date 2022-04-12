# Plots

# Imports
import torch
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple

def val_loss_vs_epoch(history, if_wandb, config): 
    
#    """
#     Function that plots the val_loss by epoch
    
#     Args: 
#         history(vector float): array containing the mse validation errors for each epoch
#         if_wandb(bool, optional): All parameters are tracked by weights and biases. Defaults to False.
#     """
   
    losses = [r['val_loss'] for r in history]
    
    # Plot
    plt.figure(figsize=(10,10))
    plt.plot(losses, '-x')
    plt.xlabel('epoch', fontsize = 14)
    plt.ylabel('val_loss', fontsize = 14)
    plt.title(f'{config.modelNN} model, with {config.learning_rate} learning_rate, {config.optimizer} optimizer & {config.epochs} epochs', fontsize = 16) 
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    
    if if_wandb:
        wandb.log({"val_loss_vs_epoch": wandb.Image(plt)})
 
    
#Plot2
def plot_pred_vs_real(x,y,g, model,if_wandb, data_type, getBM, config): 
    
#     """
#     Function that plots the prediction of the model vs the real one and calculates the total correlation coefficient and for each of the transcripts.
    
#     """
    
    # Plot
    plt.figure(figsize=(10,10))
    plt.xlabel('Predicted Values', fontsize = 14)
    plt.ylabel('Real Values', fontsize = 14)
    
    plt.title(f'{config.modelNN} model, with {config.learning_rate} learning_rate, {config.optimizer} optimizer & {config.epochs} epochs',  fontsize = 16)
    
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    
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
        
    details = {
        'gene_name' : getBM.Gene_name.to_list(),
        'trans_name' : Y_df.columns.to_list(),
        'cor_values' : cor_trans}
  
    # creating a Dataframe object 
    df = pd.DataFrame(details)
    
    if if_wandb:
        wandb.log({"plot_pred_vs_real_{}".format(data_type): wandb.Image(plt), "cor_total_{}".format(data_type): cor_total,
                   "cor_trans_{}".format(data_type): wandb.Table(dataframe=df)})
        
    PlotPredRealOutput = namedtuple('PlotPredRealOutput', ['cor_total', 'df'])
    plot_pred_real_output = PlotPredRealOutput(cor_total, df)
    
    return plot_pred_real_output
    
#Boxplot 1
def corr_vs_biotype(getBM, train_labels, df, if_wandb, data_type, config):
    
    list_biotype = getBM[getBM.Transcript_ID.isin(train_labels.columns.values)]['Biotype'].values

    data = {'trans':train_labels.columns.values,
          'corr':  df.cor_values.values,
          'biotype': list_biotype}                     
                         

    plot_df = pd.DataFrame(data)
    
    # Change default context
    sns.set_context('poster') 
    sns.set_palette('rainbow')
    
    # Plot
    plt.figure(figsize=(5,5), dpi=160) 
    ax = sns.catplot(y="biotype", x="corr",
                  data=plot_df, palette="Set3",
                orient="h", height=10, aspect=5,
                kind="violin", dodge=True, cut=0, bw=.2, legend = True)
    
    ax.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
    ax.fig.suptitle(f'{config.modelNN} model, with {config.learning_rate} learning_rate, {config.optimizer} optimizer & {config.epochs} epochs')
    
                       
    if if_wandb:
        wandb.log({"corr_vs_biotype_{}".format(data_type): wandb.Image(plt)})
#         wandb.log({f"corr_vs_biotype_{data_type} sdfsdf": wandb.Image(plt)})
    
    
def plot_results(history, scaledTrain_df, train_labels, scaled_train_gn,
                 scaledValidation_df, valid_labels, scaled_valid_gn, model, getBM,
                 if_wandb, config):

    val_loss_vs_epoch(history,if_wandb, config)
    solution_train = plot_pred_vs_real(scaledTrain_df, train_labels, scaled_train_gn,
                                       model,if_wandb, 'training', getBM, config) # training
   
    solution_val = plot_pred_vs_real(scaledValidation_df,valid_labels, scaled_valid_gn,
                                     model,if_wandb,'validation',getBM, config) # validation
    
    corr_vs_biotype(getBM, train_labels, solution_train.df, if_wandb, 'training', config)
    corr_vs_biotype(getBM, valid_labels, solution_val.df, if_wandb, 'validation', config)
    
    
    solution_train_cor_total = solution_train.cor_total
    solution_train_df = solution_train.df
    solution_val_cor_total = solution_val.cor_total
    solution_val_df = solution_val.df
    
    ReturnPlotFinalSolution = namedtuple('ReturnPlotFinalSolution', ['solution_train_cor_total', 'solution_train_df',
                                  'solution_val_cor_total', 'solution_val_df'])
    
    return_plot_final_solution = ReturnPlotFinalSolution(solution_train_cor_total, solution_train_df,
                                  solution_val_cor_total, solution_val_df)
    
    return return_plot_final_solution