# %%
# # Imports
# import pandas as pd
# import seaborn as sns
# from scipy import stats
# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import matplotlib.pyplot as plt



# %%
# Get data


# %%

# Plots
#Plot 1
def val_loss_vs_epoch(history): # Función que plotea el val_loss por epoch  
  losses = [r['val_loss'] for r in history]
  plt.plot(losses, '-x')
  plt.xlabel('epoch')
  plt.ylabel('val_loss')
  plt.title('val_loss vs. epochs');

class ReturnPlotSolution(object):
  def __init__(self, cor_total, cor_trans):
    self.cor_total =  cor_total
    self.cor_trans = cor_trans  
    
#Plot2
def plot_pred_vs_real(x,y,g, model): # Función que te plotea la predicción del 
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

  return ReturnPlotSolution(cor_total, cor_trans)

#Boxplot 1
def corr_vs_biotype(getBM, train_labels, cor_values):
  plt.figure(figsize=(15,10))
  list_biotype = getBM[getBM.Transcript_ID.isin(train_labels.columns.values)]['Biotype'].values

  data = {'trans':train_labels.columns.values,
          'corr':  cor_values,
          'biotype': list_biotype}

  plot_df = pd.DataFrame(data)
  plot_df

  # ax = sns.boxplot(x='biotype', y="corr",
  #                data=plot_df, palette="Set3")
  
  g = sns.catplot(y="biotype", x="corr",
                  data=plot_df, palette="Set3",
                orient="h", height=7, aspect=3,
                kind="violin", dodge=True, cut=0, bw=.2)


