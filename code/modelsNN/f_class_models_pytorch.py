# %%
# Imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# %% [markdown]
# # Funciones para runnear los modelos

# %%
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def do_training(model, train_loader, optimizer):
    outputs = [model.training_step(batch, optimizer) for batch in train_loader]
    return model.training_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, weights=''):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        train_epoch_end = do_training(model, train_loader, optimizer)
        # Validation phase
        val_epoch_end = evaluate(model, val_loader)
        model.epoch_end(epoch, train_epoch_end, val_epoch_end)
        history.append(val_epoch_end)
    return history

# %% [markdown]
# Linear with gene expression
# -----------------------

# %%
class DeepSF(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        # Create the weights para la expresión de los genes de las isoformas:
        self.weights_gn = torch.nn.Parameter(torch.randn(n_outputs, requires_grad=True))
        self.linear = nn.Linear(n_inputs, n_outputs)
              
    def forward(self, xb, gb):
        krn = torch.kron(torch.ones((xb.shape[0],1)), self.weights_gn) #batch size es de 32 y último 20.
        out = self.linear(xb) + krn * gb 
        return out

    def training_step(self, batch, optimizer):
        inputs, targets, gen_expr = batch 

        out = self(inputs, gen_expr) # Generate predictions
        loss = F.mse_loss(out, targets)   # Calculate loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return {'loss': loss.detach()}

    def validation_step(self, batch):
        inputs, targets, gb = batch 
        out = self(inputs, gb)                # Generate predictions
        loss_val = F.mse_loss(out, targets)       # Calculate loss

        return {'val_loss': loss_val.detach()}

    def training_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'loss': epoch_loss.item()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, loss, result):
        print("Epoch [{}], loss: {:.4f}, val_loss: {:.4f}".format(epoch, loss['loss'], result['val_loss']))

# %% [markdown]
# ## 2 hidden layers with gene expression

# %%
class DeepSF_2hidden(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.weights_gn = torch.nn.Parameter(torch.randn(n_outputs, requires_grad=True))
        self.linear1 = nn.Linear(n_inputs, 183)
        self.linear2 = nn.Linear(183, 82)
        self.linear3 = nn.Linear(82, n_outputs)
        
    def forward(self, input, gb):
        krn = torch.kron(torch.ones((input.shape[0],1)), self.weights_gn) #batch size es de 32 y último 20.
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        out = F.relu(self.linear3(x) + krn * gb)
        return out

    def training_step(self, batch, optimizer):
        inputs, targets, gen_expr = batch 
        out = self(inputs, gen_expr)   # Generate predictions
        loss = F.mse_loss(out, targets)    # Calculate loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return {'loss': loss.detach()}
    
    def validation_step(self, batch):
        inputs, targets, gb = batch 
        out = self(inputs, gb)             # Generate predictions
        loss_val = F.mse_loss(out, targets)    # Calculate loss
        return {'val_loss': loss_val.detach()}

    def training_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'loss': epoch_loss.item()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, loss, result):
        print("Epoch [{}], loss: {:.4f}, val_loss: {:.4f}".format(epoch, loss['loss'], result['val_loss']))

# %% [markdown]
# ## 2 hidden layers weighted 

# %%
#def f_rmse(out, real):
#  return torch.sum((out-real)**2)/out.nelement()

def f_rmse_weighted(input, target, weights):
    return torch.sum(weights * (input - target) ** 2)/input.nelement()

# %%
class DeepSFHiddenWeighted(nn.Module):
    def __init__(self, n_inputs, n_outputs, weights=''):
        super().__init__()

        if len(weights)>0:
          self.weights = weights

        self.weights_gn = torch.randn(n_outputs, requires_grad=True)
        self.linear1 = nn.Linear(n_inputs, 183) # 1279
        self.linear2 = nn.Linear(183, 82)
        self.linear3 = nn.Linear(82, n_outputs) # 162429
        
    def forward(self, input, gb):
        krn = torch.kron(torch.ones((input.shape[0],1)), self.weights_gn) #batch size es de 32 y último 20.
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        out = F.relu(self.linear3(x) + krn * gb)
        return out

    def training_step(self, batch, optimizer, lr):
        inputs, targets, gen_expr = batch 
        out = self(inputs, gen_expr)    # Generate predictions
        if len(weights)>0:
          loss = f_rmse_weighted(out, targets, self.weights) #F.mse_loss(out, targets)    # Calculate loss
        else:
          loss = F.mse_loss(out, targets)    # Calculate loss
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
          self.weights_gn -=  self.weights_gn.grad * lr
          self.weights_gn.grad.zero_()

        return {'loss': loss.detach()}
      
    def validation_step(self, batch):
        inputs, targets, gb = batch 
        out = self(inputs, gb)                 # Generate predictions
        if len(weights)>0:
          loss = f_rmse_weighted(out, targets, self.weights) #F.mse_loss(out, targets)    # Calculate loss
        else:
          loss = F.mse_loss(out, targets)    # Calculate loss        
        return {'val_loss': loss.detach()}

    def training_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'loss': epoch_loss.item()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, loss, result):
        print("Epoch [{}], loss: {:.4f}, val_loss: {:.4f}".format(epoch, loss['loss'], result['val_loss']))
  


