import sys

from torch_geometric.nn.conv.message_passing import MessagePassing
sys.path.insert(0,'../generate_xfoil')
import torch
from typing import List, Tuple
import numpy as np
from torch_geometric.data import Data
from libs.utils import create_edge_adjacency, pchip
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def performance_predict_dnn(model:torch.nn.Module,xss:List[float], yss:List[float],xps:List[float],yps:List[float],alpha:float,Re:float,Ncrit:float,scaler,scaler_cp,predict_cp):
    """[summary]

    Args:
        model (torch.nn.Module): [description]
        xss (List[float]): [description]
        yss (List[float]): [description]
        xps (List[float]): [description]
        yps (List[float]): [description]
        alpha (float): [description]
        Re (float): [description]
        Ncrit (float): [description]
        scaler ([type]): [description]
        scaler_cp ([type]): [description]
        predict_cp ([type]): [description]

    Returns:
        [type]: [description]
    """

    if len(xss) !=100 or len(xps) !=100:
        x = np.linspace(0,1,100) # Model uses 100 points on pressure side and suction side 
        yss = pchip(xss,yss,x)
        yps = pchip(xps,yps,x)

    y = np.concatenate((yss[0:],np.flip(yps[1:-1])))
    # Scale the data using the scaler and scaler_cp Note: This is specific to the model     
    y = scaler['y'].transform(np.array(y).reshape(-1,1))    # assume user passes in a list 
    alpha = scaler['alpha'].transform(np.array(alpha).reshape(-1,1))[0][0]
    Re = scaler['Re'].transform(np.array(Re).reshape(-1,1))[0][0]
    Ncrit = scaler['Ncrit'].transform(np.array(Ncrit).reshape(-1,1))[0][0]

    y = torch.as_tensor(y,dtype=torch.float32)
    alpha = torch.as_tensor(alpha,dtype=torch.float32)
    Re = torch.as_tensor(Re,dtype=torch.float32)
    Ncrit = torch.as_tensor(Ncrit,dtype=torch.float32)
    
    dnn_features = torch.cat((y[:,0], torch.tensor([alpha]), torch.tensor([Re]), torch.tensor([Ncrit])))

    # Create graph data object
    # Note: When predicting y, node_labels, conditions are not needed at all. These are only there to help with computing the error if they are even used at all
    dnn_features = dnn_features.to(device)
    model.to(device)
    out = model(dnn_features) # Output contains Cl, Cd, Cdp, Cm. See Step4_CreateDataset.py Line 94 where data_y is defined
    out = out.cpu().numpy()
    Cl = float(scaler['Cl'].inverse_transform(out[0].reshape(-1,1))[0,0])
    Cd = float(scaler['Cd'].inverse_transform(out[1].reshape(-1,1))[0,0])
    Cdp = float(scaler['Cdp'].inverse_transform(out[2].reshape(-1,1))[0,0])
    Cm = float(scaler['Cm'].inverse_transform(out[3].reshape(-1,1))[0,0])

    Cp_ss = list()
    Cp_ps = list()
    # ! Need to debug this part
    if predict_cp:
        Cp_points = len(out)-4

        Cp_ss = out[4:4+int(Cp_points/2)+1]
        Cp_ps = np.append(out[4+int(Cp_points/2):],out[4])
        Cp_ps = np.flip(Cp_ps)
            
        for i in range(len(scaler_cp)):                
            Cp_ss[i] = scaler_cp[i].inverse_transform(Cp_ss[i].reshape(-1,1))[0] # Transform Cp for each value of x
        for i in range(len(scaler_cp)):
            Cp_ps[i] = scaler_cp[i].inverse_transform(Cp_ps[i].reshape(-1,1))[0] # Transform Cp for each value of x

    return Cl,Cd,Cdp,Cm, Cp_ss,Cp_ps

@torch.no_grad()
def performance_predict_gnn(model:MessagePassing,xss:List[float],yss:List[float],xps:List[float],yps:List[float],alpha:float,Re:float,Ncrit:float,scaler,scaler_cp,predict_cp):
    """Predicts the performance using Graph Neural Networks. If you were to make an API, you would need to wrap this function with another one that loads all the models

    Args:
        model (MessagePassing): Graph neural network model 
        xss (List[float]): List of points that describe the suction side x coordinate
        yss (List[float]): List of points that describe the suction side y coordinate
        xps (List[float]): List of points that describe the pressure side x coordinate
        yps (List[float]): List of points that describe the pressure side y coordinate
        alpha (float): angle of attack
        Re (float): Reynolds number 
        Ncrit (float): measurement of free flow turbulence http://websites.umich.edu/~mdolaboratory/pdf/Shi2018a.pdf
        scaler ([type]): [description]
        scaler_cp ([type]): [description]
    """
    if len(xss) !=100 or len(xps) !=100:
        x = np.linspace(0,1,100) # Model uses 100 points on pressure side and suction side 
        yss = pchip(xss,yss,x)
        yps = pchip(xps,yps,x)

    x = np.concatenate((xss[0:],np.flip(xps[1:-1]))) # This is already in 0 to 1
    y = np.concatenate((yss[0:],np.flip(yps[1:-1]))) # 
    
    # Scale the data using the scaler and scaler_cp Note: This is specific to the model 
    # y = scaler['y'].transform(np.array(y).reshape(-1,1))    # assume user passes in a list 
    alpha = scaler['alpha'].transform(np.array(alpha).reshape(-1,1))[0][0]
    Re = scaler['Re'].transform(np.array(Re).reshape(-1,1))[0][0]
    Ncrit = scaler['Ncrit'].transform(np.array(Ncrit).reshape(-1,1))[0][0]
    
    x = torch.tensor(x.reshape(-1,1), dtype=torch.float32)
    y = torch.as_tensor(y.reshape(-1,1), dtype=torch.float32)
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    Re = torch.as_tensor(Re, dtype=torch.float32)
    Ncrit = torch.as_tensor(Ncrit, dtype=torch.float32)

    features = torch.zeros((y.shape[0],3))
    features[:,0] = alpha
    features[:,1] = Re
    features[:,2] = Ncrit
    edge_index = create_edge_adjacency(len(y))
    edge_index = np.array(edge_index) # Edge Adjacency 
    if (edge_index.shape[0]!=2):
        edge_index = edge_index.transpose()
    edge_index = torch.as_tensor(edge_index,dtype=torch.long).contiguous()

    pos = torch.cat((x,y),axis=1)
    edge_attr = torch.ones((edge_index.shape[1],pos.shape[1]),dtype=torch.float32)

    # Create graph data object
    # Note: When predicting y, node_labels, conditions are not needed at all. These are only there to help with computing the error if they are even used at all
    data = Data(x=features,edge_index=edge_index,pos=pos,y=None,node_labels=None,conditions=None,edge_attr=edge_attr) 
    data.batch = [0]
    
    data=data.to(device)
    model.to(device)
    out = model(data) # Output contains Cl, Cd, Cdp, Cm. See Step4_CreateDataset.py Line 94 where data_y is defined
    out = out.cpu().numpy()[0]
    Cl = float(scaler['Cl'].inverse_transform(out[0].reshape(-1,1))[0,0])
    Cd = float(scaler['Cd'].inverse_transform(out[1].reshape(-1,1))[0,0])
    Cdp = float(scaler['Cdp'].inverse_transform(out[2].reshape(-1,1))[0,0])
    Cm = float(scaler['Cm'].inverse_transform(out[3].reshape(-1,1))[0,0])

    Cp_ss = list()
    Cp_ps = list()
    # ! Need to debug this part
    if predict_cp:
        Cp_points = len(out)-4

        Cp_ss = out[4:4+int(Cp_points/2)+1]
        Cp_ps = np.append(out[4+int(Cp_points/2):],out[4])
        Cp_ps = np.flip(Cp_ps)
            
        for i in range(len(scaler_cp)):                
            Cp_ss[i] = scaler_cp[i].inverse_transform(Cp_ss[i].reshape(-1,1))[0] # Transform Cp for each value of x
        for i in range(len(scaler_cp)):
            Cp_ps[i] = scaler_cp[i].inverse_transform(Cp_ps[i].reshape(-1,1))[0] # Transform Cp for each value of x

    return Cl,Cd,Cdp,Cm,Cp_ss,Cp_ps
