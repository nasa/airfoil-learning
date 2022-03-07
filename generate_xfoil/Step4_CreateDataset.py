import pickle
from typing import Dict, List, Tuple
from tqdm import trange
import numpy as np
import random, json
import torch, glob, os
import os.path as osp
from torch.utils.data import random_split
import torch_geometric.transforms as T
from libs.utils import create_edge_adjacency
from torch_geometric.data import Data
import sys
from libs.utils import pchip

sys.path.insert(0,'libs')

def shuffle_and_save(scaled_data: List, process_path:str,file_prefix:str,train_test_split:float=0.7):
    """Shuffle the list and save

    Args:
        scaled_data (List): [description]
        file_prefix (str): [description]
        train_test_split (float, optional): [description]. Defaults to 0.7.
    """
    # Load all the designs 
    random.shuffle(scaled_data) # Shuffle the list
    
    train_size = int(len(scaled_data)*train_test_split)
    test_size = len(scaled_data) - train_size     

    train_subset, test_subset = random_split(scaled_data,[train_size, test_size])
    train_dataset = [scaled_data[i] for i in train_subset.indices] 
    test_dataset = [scaled_data[i] for i in test_subset.indices]     

    torch.save(train_dataset,os.path.join(process_path,f'{file_prefix}_train.pt'))
    torch.save(test_dataset,os.path.join(process_path,f'{file_prefix}_test.pt'))

def CreateDatasetFromJson(airfoil:Dict,scaler:Dict,scaler_cp:Dict,cp_points:int) -> Tuple[List[Data], List[Data], List[Data], List[Data]]:
    """Takes a single json file and creates a tuple containing lists of graph data objects and also deep neural network data objects. These objects are combined together and later used by pytorch dataloader 

    Args:
        airfoil (Dict): Dictionary containing properties of the airfoil0
        scaler (Dict): Dictionary containing normalization parameters 
        scaler_cp (Dict): Dictionary containing normalization parameters for Cp

    Returns:
        Tuple containing:
            List[Data], List[Data], List[Data], List[Data]]: [description]
    """
    
    '''
        Normalize the x and the y for airfoil 
    '''
    xss = airfoil['xss']        
    yss = airfoil['yss']

    xps = airfoil['xps']
    yps = airfoil['yps']
    
    x = np.concatenate((xss[0:],np.flip(xps[1:-1]))).reshape(-1,1) # This is already in 0 to 1
    y = np.concatenate((yss[0:],np.flip(yps[1:-1]))).reshape(-1,1) # 
    y_scaled = scaler['y'].transform(y) # Do not transform y for gnn. This is for DNN only 
    edge_index = create_edge_adjacency(len(x))

    graph_scaled_data = list()
    graph_scaled_data_cp = list() 
    dnn_scaled = list() 
    dnn_scaled_cp = list()

    for p in range(len(airfoil['polars'])):            
        polar = airfoil['polars'][p]

        Cp_ss = np.array(polar['Cp_ss'])
        Cp_ps = np.array(polar['Cp_ps'])
                    
        alpha = scaler['alpha'].transform(np.array(polar['alpha']).reshape(-1,1))[0][0]
        Re = scaler['Re'].transform(np.array(polar['Re']).reshape(-1,1))[0][0]
        Ncrit = scaler['Ncrit'].transform(np.array(polar['Ncrit']).reshape(-1,1))[0][0]

        # Normalize Cl, Cd, Cdp, Cm
        Cl = scaler['Cl'].transform(np.array(polar['Cl']).reshape(-1,1))
        Cd = scaler['Cd'].transform(np.array(polar['Cd']).reshape(-1,1))
        Cdp = scaler['Cdp'].transform(np.array(polar['Cdp']).reshape(-1,1))
        Cm = scaler['Cm'].transform(np.array(polar['Cm']).reshape(-1,1))

        # Scale Cp
        Cp = np.concatenate(( Cp_ss, np.flip(Cp_ps[1:-1]) ))
        Cp = torch.as_tensor(scaler['Cp'].transform(Cp.reshape(-1,1))[0:],dtype=torch.float32) # This has been normalized as a whole

        data_y = torch.as_tensor(np.hstack([ Cl, Cd, Cdp, Cm ]), dtype=torch.float32)[0]
        edge_index = np.array(edge_index) # Edge Adjacency 
        if (edge_index.shape[0]!=2):
            edge_index = edge_index.transpose()
        edge_index = torch.as_tensor(edge_index,dtype=torch.long).contiguous()

        x = torch.as_tensor(np.hstack([x]), dtype=torch.float32)
        y = torch.as_tensor(np.hstack([y]), dtype=torch.float32)
        y_scaled = torch.as_tensor(np.hstack([y_scaled]), dtype=torch.float32)
        conditions=torch.as_tensor(np.hstack([alpha, Re, Ncrit]),dtype=torch.float32)
        pos = torch.as_tensor(np.hstack([x, y]), dtype=torch.float32)
        edge_attr = torch.ones((edge_index.shape[1],pos.shape[1]),dtype=torch.float32)
        '''
            airfoil with all values scaled by global min/max or mean/std 
        '''
        # d = Data(x=data_x,edge_index=edge_index,pos=pos,y=data_y,node_labels=Cp,conditions=conditions)            
        features = torch.zeros((y.shape[0],3))
        # features[:,0] = x[:,0]
        # features[:,1] = y[:,0]
        features[:,0] = alpha
        features[:,1] = Re
        features[:,2] = Ncrit
        # scaled_data
        graph_scaled_data.append(Data(x=features,edge_index=edge_index,pos=pos,y=data_y,node_labels=Cp,conditions=conditions,edge_attr=edge_attr))

        '''
            airfoil with all values except for cp scaled by global min/max or mean/std 
        '''
        Cp_ss_scaled = Cp_ss
        Cp_ps_scaled = Cp_ps 
        for i in range(len(scaler_cp)):                
            Cp_ss_scaled[i] = scaler_cp[i].transform(Cp_ss[i].reshape(-1,1))[0] # Transform Cp for each value of x
        for i in range(len(scaler_cp)):
            Cp_ps_scaled[i] = scaler_cp[i].transform(Cp_ps[i].reshape(-1,1))[0] # Transform Cp for each value of x
        Cp_ps_scaled = np.flip(Cp_ps[1:-1])
        Cp_scaled = np.concatenate(( Cp_ss_scaled, Cp_ps_scaled ))
        Cp_scaled = torch.as_tensor(Cp_scaled.reshape(-1,1)[0:],dtype=torch.float32)
        
        # scaled_data_cp
        graph_scaled_data_cp.append(Data(x=features,edge_index=edge_index,pos=pos,y=data_y,node_labels=Cp_scaled,conditions=conditions,edge_attr=edge_attr))

        '''
            Deep Neural Network 
        '''
        dnn_features = (torch.cat((y_scaled[:,0], torch.tensor([alpha]), torch.tensor([Re]), torch.tensor([Ncrit])))).float()
        dnn_labels = (torch.cat((data_y,Cp[:,0])))    
        dnn_labels_cp = (torch.cat((data_y,Cp_scaled[:,0])))     

        dnn_scaled.append((dnn_features,dnn_labels))
        dnn_scaled_cp.append((dnn_features,dnn_labels_cp))

    return graph_scaled_data, graph_scaled_data_cp, dnn_scaled, dnn_scaled_cp

def CreateDataset(data_folder:str='json',processed_path:str='datasets',
                        use_standard_scaler:bool=True):
    """Create a dataset that can be used to train a graph neural network 

    Reference:
        https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html

    Args:
        data_folder (str, optional): name of file to be scraped . Defaults to 'json'.
        processed_path (str, optional): path to save the pytorch dataset. Defaults to 'datasets'.
        use_standard_scaler (bool, optional): Whether to use standard scaler or min_max. Defaults to True.

    Returns:
        Saves 4 files in the processed_path folder 
            graph_scaled_data.pt: Graph Data format with cp all scaled by a common scaler
            graph_scaled_data_cp.pt: Graph Data format with cp individually scaled at each x value
            dnn_scaled.pt: Deep neural format with cp all scaled by a common scaler
            dnn_scaled_cp.pt: Deep neural network format  with cp individually scaled at each x value

    """
    os.makedirs(processed_path,exist_ok=True)
    
    data_files = glob.glob(osp.join(data_folder,'*.json'))
    jsons = list() 
    for filename in data_files:
        with open(filename,'r') as f:
            jsons.append(json.load(f))
    with open('scalers.pickle','rb') as f:
        data = pickle.load(f)
        if use_standard_scaler:
            scaler = data['standard']
            scaler_cp = data['standard_cp']
        else:
            scaler = data['min_max']
            scaler_cp = data['min_max_cp']
    
    graph_scaled_data = list() # All airfoil parameters are scaled by the global min and max or mean and standard dev
    graph_scaled_data_cp = list() # All except for Cp is scaled by global min and max. Cp is scaled at each x
    dnn_scaled = list()
    dnn_scaled_cp = list()

    pbar = trange(len(jsons),desc='Processing')
    for c in pbar:
        out1, out2, out3, out4 = CreateDatasetFromJson(jsons[c],scaler,scaler_cp,50)
        pbar.desc="Extending List"
        graph_scaled_data.extend(out1)
        graph_scaled_data_cp.extend(out2)
        dnn_scaled.extend(out3)
        dnn_scaled_cp.extend(out4)
        pbar.desc="Processing"
        
    shuffle_and_save(graph_scaled_data,processed_path,'graph_scaled_data',0.7)
    shuffle_and_save(graph_scaled_data_cp,processed_path,'graph_scaled_data_cp',0.7)
    shuffle_and_save(dnn_scaled,processed_path,'dnn_scaled_data',0.7)
    shuffle_and_save(dnn_scaled_cp,processed_path,'dnn_scaled_data_cp',0.7)


if __name__ == "__main__":
    CreateDataset(data_folder='json_cp_resize',processed_path='datasets/standard/',use_standard_scaler=True)
    CreateDataset(data_folder='json_cp_resize',processed_path='datasets/minmax/',use_standard_scaler=False)
    # transform_test_train()