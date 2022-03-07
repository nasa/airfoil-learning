from pathlib import Path
from typing import List
import torch
import json
from sklearn.model_selection import train_test_split 
import json
import os.path as osp

def setup_dataset(train_filename:str,test_filename:str,percent_dataset:float=1,train_percentage:float=0.8,train_indices:List[int]=None, test_indices:List[int]=None,scaler_type:str="minmax"):
    """Setup the Train and Validation datasets

    Args:
        train_filename (str): [description]
        test_filename (str): [description]
        percent_dataset (float, optional): percentage of dataset to use 0 to 1. Defaults to 1.
        train_indices (List[int], optional): array indices to use for training. Defaults to None.
        test_indices (List[int], optional): array indices to use for test. Defaults to None.
        scaler_type (str, optional): string to describe the type of scaler to use. This determines where the dataset is saved. Defaults to "minmax".

    Returns:
        (tuple): tuple containing:

            - **train_dataset** (torch.utils.data.TensorDataset): pitch distribution
            - **val_dataset** (torch.utils.data.TensorDataset): pitch to chord distribution

    """

    train_dataset = torch.load(train_filename)
    test_dataset = torch.load(test_filename)
    
    # Combined train and test
    train_dataset.extend(test_dataset)
    
    if (not train_indices):
        all_data = range(len(train_dataset))
        # Shuffle the dataset and select a percent from it 
        new_dataset,_ = train_test_split(all_data, test_size=1-percent_dataset, train_size=percent_dataset)
        train_indices, test_indices = train_test_split(new_dataset, test_size=1-train_percentage, train_size=train_percentage)
    
    test_dataset = [train_dataset[t] for t in test_indices]
    train_dataset = [train_dataset[t] for t in train_indices]
    
    Path(osp.join('local_dataset',scaler_type)).mkdir(parents=True, exist_ok=True)
    Path(osp.join('local_dataset',scaler_type)).mkdir(parents=True, exist_ok=True)
    if 'graph' in train_filename:
        data_params = {'input_size':train_dataset[0].x.shape[0],'output_size':len(train_dataset[0].y),'node_labels':train_dataset[0].node_labels.shape[0]}
        with open(osp.join('local_dataset',scaler_type,'gnn_dataset_properties.json'), 'w') as outfile:
            json.dump(data_params, outfile)
        torch.save(train_dataset,osp.join('local_dataset',scaler_type,'train_gnn.pt'))
        torch.save(test_dataset,osp.join('local_dataset',scaler_type,'test_gnn.pt'))
    else:
        features = train_dataset[0][0]
        labels = train_dataset[0][1]
        data_params = {'input_size':len(features),'output_size':len(labels)}
        with open(osp.join('local_dataset',scaler_type,'dnn_dataset_properties.json'), 'w') as outfile:
            json.dump(data_params, outfile)
        torch.save(train_dataset,osp.join('local_dataset',scaler_type,'train_dnn.pt'))
        torch.save(test_dataset,osp.join('local_dataset',scaler_type,'test_dnn.pt'))
    
    return train_indices, test_indices

if __name__ == '__main__':
    with open('settings.json', 'r') as infile:    # Loads the settings for all networks
        settings = json.load(infile)
    
    percent_train = -1 
    for args in settings['data']:
        if percent_train == args["percent_train"]:
            train_indices, test_indices = setup_dataset(args['train_filename'], args['test_filename'], args['percent_dataset'], args["percent_train"],train_indices,test_indices,args['scaler_type'])
        else:
            train_indices, test_indices = setup_dataset(args['train_filename'], args['test_filename'], args['percent_dataset'], args["percent_train"],None,None,args['scaler_type'])
        percent_train = args["percent_train"]
    print('local_dataset created')