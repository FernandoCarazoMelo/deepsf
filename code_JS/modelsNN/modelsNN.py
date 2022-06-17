# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

class DeepSF(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        # Create the weights for the gene expression of theisoforms:
        self.weights_gn = torch.nn.Parameter(torch.randn(n_outputs, requires_grad=True))
        self.linear = nn.Linear(n_inputs, n_outputs)
              
    def forward(self, xb, gb):
        krn = torch.kron(torch.ones((xb.shape[0],1)), self.weights_gn)
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

class DeepSF_2hidden(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.weights_gn = torch.nn.Parameter(torch.randn(n_outputs, requires_grad=True))
        
        self.linear1 = nn.Linear(n_inputs, 183)
        self.bn1 = nn.BatchNorm1d(183)
        self.linear2 = nn.Linear(183, 82)
        self.bn2 = nn.BatchNorm1d(82)
        self.linear3 = nn.Linear(82, n_outputs)
        
    def forward(self, xb, gb):
        krn = torch.kron(torch.ones((xb.shape[0],1)), self.weights_gn)
        x = F.relu(self.bn1(self.linear1(xb)))
        x = F.relu(self.bn2(self.linear2(x)))
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
        
class DeepSFHiddenWeighted(nn.Module):
    def __init__(self, n_inputs, n_outputs, weights=''):
        super().__init__()

        if len(weights)>0:
            self.weights = weights

        self.weights_gn = torch.randn(n_outputs, requires_grad=True)
        self.linear1 = nn.Linear(n_inputs, 183) 
        self.linear2 = nn.Linear(183, 82)
        self.linear3 = nn.Linear(82, n_outputs)
        
    def forward(self, input, gb):
        krn = torch.kron(torch.ones((input.shape[0],1)), self.weights_gn)
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
        
         
class DeepAE(nn.Module):    
    def __init__(self, n_inputs):
        super().__init__()
        self.linear1 = nn.Linear(n_inputs, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, n_inputs)
    
    def forward(self, inputs):
        x = F.sigmoid(self.bn1(self.linear1(inputs)))
        x = F.sigmoid(self.bn2(self.linear2(x)))
        x = F.sigmoid(self.linear3(x))
        out = F.sigmoid(self.linear4(x))
        return out
    
    def training_step(self, batch, optimizer):
        inputs, targets = batch 
        out = self(inputs) # Generate predictions
        loss = F.mse_loss(out, targets) # Calculate loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return {'loss': loss.detach()}
    
    def validation_step(self, batch):
        inputs, targets = batch 
        out = self(inputs) # Generate predictions
        loss_val = F.mse_loss(out, targets) # Calculate loss
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

class DeepSF_AE_Ensemble(nn.Module):
    def __init__(self, n_inputs, n_outputs, modelAE):
        super().__init__()

        self.modelAE = modelAE
        self.weights_gn = torch.nn.Parameter(torch.randn(n_outputs, requires_grad=True))

        self.linear1 = nn.Linear(n_inputs, 183)
        self.bn1 = nn.BatchNorm1d(183)
        self.linear2 = nn.Linear(183, 82)
        self.bn2 = nn.BatchNorm1d(82)
        self.linear3 = nn.Linear(82+512, n_outputs)

    def forward(self, xb, gb, xa):
        krn = torch.kron(torch.ones((xb.shape[0],1)), self.weights_gn)
        x = F.relu(self.bn1(self.linear1(xb)))
        x = F.relu(self.bn2(self.linear2(x)))

        x = torch.cat((x, self.modelAE(xa).detach()), dim=1) #Detaching x2, so modelAE wont be updated 
        out = F.relu(self.linear3(x) + krn * gb)
       
        return out

    def training_step(self, batch, optimizer):
        inputs_sf, targets_sf, gen_expr, inputs_ae = batch 
        out = self(inputs_sf, gen_expr, inputs_ae) # Generate predictions
        loss = F.mse_loss(out, targets_sf)    # Calculate loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return {'loss': loss.detach()}

    def validation_step(self, batch):
        inputs_sf, targets_sf, gen_expr, inputs_ae = batch 

        out = self(inputs_sf, gen_expr, inputs_ae) # Generate predictions
        loss_val = F.mse_loss(out, targets_sf)    # Calculate loss

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
