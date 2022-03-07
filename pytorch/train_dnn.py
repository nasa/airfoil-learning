import os.path as osp
from typing import Dict 
import torch
from torch_geometric.loader import DataLoader
import argparse
import logging, datetime
from tqdm import tqdm, trange
from gnn_model import GnnModel
from MultiLayerLinear import MultiLayerLinear
import torch.nn.functional as F
import json 
import pandas as pd
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(train_dl:DataLoader,optimizer:torch.optim.Optimizer, model:torch.nn.Module,predict_cp:bool) -> float:
    """Trains a deep neural network for one epoch. This is meant to be called in a loop

    Args:
        train_dl (DataLoader): Dataloader containing a list of datasets 
        optimizer (torch.optim.Optimizer): Optimizer
        model (torch.nn.Module): Graph neural network model
        output_channels (int): Number of output channels

    Returns:
        float: network loss 
    """
    model.train()
    total_loss = 0
    n = 0
    pbar = tqdm(train_dl)
    accumulation_steps = 10
    i = 0
    model.zero_grad()
    for _, data in enumerate(pbar):
        features, labels = data
        batch_size = features.shape[0]
        features = features.float().to(device)
        if predict_cp:
            labels = labels.float().to(device)
        else:
            labels = labels[:,0:4].float().to(device)
        
        out = model(features)        
        loss = F.mse_loss(out, labels)
        loss = loss / accumulation_steps                # Normalize our loss (if averaged)
        loss.backward()

        total_loss += loss.item() * batch_size # sum of all loss for all data in batch
        n += batch_size
        # Performance improvements taken from https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/ and https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            model.zero_grad()                           # Reset gradients tensors        
            pbar.set_description(f"Train Mode: Loss {loss:05e}")
        i+=1
    return total_loss / n


def evaluate(test_dl:DataLoader,model:torch.nn.Module,predict_cp:bool):
    """Evaluates the graph neural network model on a test dataset 

    Args:
        dataloader (DataLoader): This is intended to be the test dataloader
        model (torch.nn.Module): this is the neural network model for a graph network. 
        output_channels (int): number of output channels

    Returns:
        float: Loss
    """
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        pbar = tqdm(test_dl)
        for _, data in enumerate(pbar):
            features, labels = data
            features = features.float().to(device)
            if predict_cp:
                labels = labels.float().to(device)
            else:
                labels = labels[:,0:4].float().to(device)            
            out = model(features)        

            batch_loss = F.mse_loss(out, labels)
            total_loss += batch_loss.item() * features.shape[0]            
            n += features.shape[0]     
            pbar.set_description(f"Test Mode: Loss {batch_loss:05e}")
    return total_loss / n


def save_checkpoint(output_dir:str, state:Dict, model_name:str):    
    """Saves the model checkpoint to a file 

    Args:
        output_dir (str): Directory to save the checkpoint
        state (Dict): This contains information needed to resume training. {'state_dict':Model_state/variables, 'optimizer':optimizer state, 'epoch': last epoch run}
        model_name (str): Name describing the model 
    """
    os.makedirs(output_dir,exist_ok=True)
    filename = osp.join(output_dir,model_name + '_checkpoint.pt.tar')
    torch.save(state, filename)

def load_checkpoint(output_dir:str, model_name:str):
    """Loads a checkpoint for a given model 

    Args:
        output_dir (str): Directory where to look for saved checkpoints 
        model_name (str): name of the model

    Returns:
        Dict: dictionary describing the state
    """

    filename = osp.join(output_dir,model_name + '_checkpoint.pt.tar')
    if osp.exists(filename):
        state = torch.load(filename)
        return state
    else:
        return None

if __name__=="__main__":
    """Main code execution. Structuring this way allows the use of multiple workers
    """
    
    with open('settings.json', 'r') as infile:          # Loads the settings for the graph neural network 
        args = json.load(infile)
        dataset_type = 'dnn'
        gnn_networks = [a for a in args['networks'] if a['type'] == dataset_type]
        for args in gnn_networks:
            data_folder = osp.join('local_dataset',args['scaler_type'])
            
            with open(osp.join(data_folder,f'{dataset_type}_dataset_properties.json'), 'r') as infile:
                data_params = json.load(infile)

            train_dataset = torch.load(osp.join(data_folder, 'train_dnn.pt'))
            test_dataset = torch.load(osp.join(data_folder, 'test_dnn.pt'))

            train_args = args['train']
            train_dl = DataLoader(train_dataset, shuffle=True, batch_size=train_args['batch_size'],pin_memory=True)
            test_dl = DataLoader(test_dataset, shuffle=False, batch_size=train_args['batch_size'],pin_memory=True)

            train_args['input_size'] = data_params['input_size']
            
            if train_args['predict_cp']:
                train_args['output_size'] = data_params['output_size']
                model = MultiLayerLinear(in_channels=data_params['input_size'],out_channels=train_args['output_size'],h_sizes=train_args['Layers'])                
            else:
                train_args['output_size'] = 4
                model = MultiLayerLinear(in_channels=data_params['input_size'],out_channels=train_args['output_size'],h_sizes=train_args['Layers'])
            
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=train_args['learning_rate'],weight_decay=train_args['weighted_decay'])
            model_name = str(model)
            checkpoint_data = load_checkpoint(train_args['output_dir'], model_name)
            if (checkpoint_data):
                    model.load_state_dict(checkpoint_data['state_dict'])     
                    optimizer.load_state_dict(checkpoint_data['optimizer'])
                    start_epoch = checkpoint_data['epoch'] +1
                    logging.info("Loaded model state starting epoch" + str(start_epoch))
            else:
                start_epoch = 1

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)

            loss_track = list()
            lr_history = list()
            for epoch in trange(start_epoch, train_args['epochs']):
                loss = train_one_epoch(train_dl,optimizer,model,train_args['predict_cp'])
                if epoch % train_args['validation_epoch'] == 0:
                    val_loss = evaluate(test_dl,model,train_args['predict_cp'])
                    logging.info(f"Validation: Epoch {epoch:03d} Loss {loss:05e}")
                    loss_track.append({'model_name':str(model),'timestamp':datetime.datetime.now(),'epoch':epoch,'loss':loss,'validation_loss':val_loss}) 
                else:
                    loss_track.append({'model_name':str(model),'timestamp':datetime.datetime.now(),'epoch':epoch,'loss':loss,'validation_loss':-1}) # This means there's is no test loss saved

                scheduler.step()
                lr_history.append({'lr': get_lr(optimizer), 'epoch':epoch})

                print(f"Training: Epoch {epoch:03d} Loss {loss:05e}")
                logging.info(f"Training: Epoch {epoch:03d} Loss {loss:05e}")
                
                save_checkpoint(train_args['output_dir'],{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss_track': loss_track,
                        'LR_history':lr_history,
                        'parameters':train_args
                    }, model_name) 