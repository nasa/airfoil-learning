'''
    This code reads a checkpoint file and reads from the test dataset to plot 
'''
import sys
import torch 
import matplotlib.pyplot as plt
from labellines import labelLines
import os.path as osp
import glob
from MultiLayerLinear import MultiLayerLinear
from gnn_model import GnnModel
from train_gnn import get_lr
import pandas as pd
import matplotlib.pylab as pylab
'''
    Compare the checkpoints found here. 

    Note: 
        I did some rearranging of the files so that they were in separate folders 
'''
# These 2 will be compared
dnn_no_cp = glob.glob('checkpoints_dnn_no_cp' + "/**/*.pt.tar", recursive = True)
gnn_no_cp = glob.glob('checkpoints_gnn_no_cp' + "/**/*.pt.tar", recursive = True)

# These 2 will be compared
dnn_cp = glob.glob('checkpoints_dnn_cp' + "/**/*.pt.tar", recursive = True)
gnn_cp = glob.glob('checkpoints_gnn_cp' + "/**/*.pt.tar", recursive = True)


def plot_history_dnn(model_histories,title_prefix,ylim_train=[8E-5,1E-2],ylim_test=[8E-5,1E-2]):
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(20,8),dpi=300,num=1,clear=True) # Learning Rate with Loss 
    ax1 = fig.add_subplot(121) # Train Loss    
    ax1.set_title(f'{title_prefix} Training Loss vs. Epoch')
    ax1.set_yscale('log')    
    ax1.grid()
    ax1.set_ylim(ylim_train)
    ax1.set_ylabel('Loss',fontsize=18)
    ax1.set_xlabel('Epoch',fontsize=18)
    
    ax11 = ax1.twinx()    
    ax11.set_yscale('log')

    ax2 = fig.add_subplot(122) # Test Loss
    ax2.set_title(f'{title_prefix} Test Loss vs. Epoch')
    ax2.set_yscale('log')  
    ax2.grid()
    ax2.set_ylim(ylim_test)
    ax2.set_ylabel('Loss',fontsize=18)
    ax2.set_xlabel('Epoch',fontsize=18)
    for history in model_histories:
        history = torch.load(history)
        loss_track = pd.DataFrame(history['loss_track'])
        lr_history = pd.DataFrame(history['LR_history'])
        train_args = history['parameters']
        model_name = loss_track['model_name'][0]
        val_loss = loss_track[loss_track['validation_loss']>0]
        layer_size = train_args['Layers'][0]
        n_layers = len(train_args['Layers'])
        if 'minmax' in train_args['output_dir']:
            label = f'minmax-MLP-{layer_size}x{n_layers}'
        else:
            label = f'standard-MLP-{layer_size}x{n_layers}'

        ax1.plot(loss_track['epoch'],loss_track['loss'],'-',linewidth=1.5,label=label)
        ax2.plot(val_loss['epoch'],val_loss['validation_loss'],'-',linewidth=1.5,label=label)

    ax11.plot(lr_history['epoch'],lr_history['lr'],'-',color='darkorchid',linewidth=1.5, label='Learning Rate')
    labelLines(ax11.get_lines(), zorder=2.5)


    ax1.legend(loc="upper right",fontsize=14)
    ax2.legend(loc="upper right",fontsize=14)
    fig.tight_layout(pad=3.0)
    title_prefix = title_prefix.replace(' ','_')
    plt.savefig(f'Loss_History_dnn_{title_prefix}.png')

def plot_history_gnn(model_histories,title_prefix,ylim_train=[8E-5,1E-2],ylim_test=[8E-5,1E-2]):
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(20,8),dpi=300,num=1,clear=True) # Learning Rate with Loss 
    ax1 = fig.add_subplot(121) # Train Loss
    
    ax1.set_title(f'{title_prefix} Training Loss vs. Epoch')
    ax1.set_yscale('log')    
    ax1.grid()
    ax1.set_ylim(ylim_train)
    ax1.set_ylabel('Loss',fontsize=18)
    ax1.set_xlabel('Epoch',fontsize=18)

    ax11 = ax1.twinx()    
    ax11.set_yscale('log')

    ax2 = fig.add_subplot(122) # Test Loss

    ax2.set_title(f'{title_prefix} Test Loss vs. Epoch')
    ax2.set_yscale('log')  
    ax2.grid()
    ax2.set_ylim(ylim_test)
    ax2.set_ylabel('Loss',fontsize=18)
    ax2.set_xlabel('Epoch',fontsize=18)
    for history in model_histories:
        history = torch.load(history)
        loss_track = pd.DataFrame(history['loss_track'])
        lr_history = pd.DataFrame(history['LR_history'])
        train_args = history['parameters']
        model_name = loss_track['model_name'][0]
        val_loss = loss_track[loss_track['validation_loss']>0]        

        if train_args['hiddenLayers']:
            layer_size = train_args['hiddenLayers'][0]
            n_layers = len(train_args['hiddenLayers'])
            label=f'GNN-MLP-{layer_size}x{n_layers}'
        else:
            label='GNN-MLP-None'
        
        if 'minmax' in train_args['output_dir']:
            label = 'minmax-'+label
        else:
            label = 'standard-'+label
        ax1.plot(loss_track['epoch'],loss_track['loss'],'-',linewidth=1.5,label=label)
        ax2.plot(val_loss['epoch'],val_loss['validation_loss'],'-',linewidth=1.5,label=label)

    ax11.plot(lr_history['epoch'],lr_history['lr'],'-',color='darkorchid',linewidth=1.5, label='Learning Rate')
    labelLines(ax11.get_lines(), zorder=2.5)


    ax1.legend(loc="upper right",fontsize=14)
    ax2.legend(loc="upper right",fontsize=14)
    fig.tight_layout(pad=3.0)
    title_prefix = title_prefix.replace(' ','_')
    plt.savefig(f'Loss_History_gnn_{title_prefix}.png')

def cleanup(model_histories):
    for filename in model_histories:
        history = torch.load(filename)
        loss_track = history['loss_track']
        lr_history = history['LR_history']
        train_args = history['parameters']
        # this is only here if you mess up the saving of learning rate. You can recreate it. It takes a while to train, this is faster
        if 'GnnLayers' in train_args:         
            lr_step = 2
            GnnLayers = train_args['GnnLayers']
            linear_layer = MultiLayerLinear(in_channels=train_args['input_size']*GnnLayers[1],out_channels=train_args['output_size'],h_sizes=train_args['hiddenLayers'])
            model = GnnModel(train_args['input_size'],GnnLayers,linear_layers=linear_layer)
        else:
            lr_step=10            
            model = MultiLayerLinear(in_channels=train_args['input_size'],out_channels=train_args['output_size'],h_sizes=train_args['Layers'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_args['learning_rate'],weight_decay=train_args['weighted_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5, verbose=False)
        lr_history = list()
        for epoch in range(train_args['epochs']):
            scheduler.step()
            lr_history.append({'lr': get_lr(optimizer), 'epoch':epoch})

        loss_track2 = list()
        for l in loss_track:
            loss_track2.append({'model_name':l[0],'timestamp':l[1],'epoch':l[2],'loss':l[3],'validation_loss':l[4]})

        history['LR_history'] = lr_history
        history['loss_track'] = loss_track2
        torch.save(history,filename)

if __name__=="__main__":
    plot_history_dnn(dnn_no_cp,'No Cp',[8E-5,1E-3],[1E-4,1E-2])    
    plot_history_dnn(dnn_cp,'Cp',[0.1,0.3],[2.4,2.8])
    plot_history_gnn(gnn_no_cp,'No Cp',[5E-5,0],[1e-4,0.3])
    plot_history_gnn(gnn_cp,'Cp',[2,4],[2,4])
    
