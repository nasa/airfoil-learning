import torch
from sklearn import preprocessing
import pickle
import torch
import os.path as osp
import sys
# sys.path.insert(0,'../create_datasets')
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
import torch

def convert_to_ndarray(t):
    """
        converts a scalar or list to numpy array 
    """
    if (type(t) is torch.Tensor):
        t = t.numpy()
    else:
        if type(t) is not np.ndarray and type(t) is not list: # Scalar
            t = np.array([t],dtype=float)
        elif (type(t) is list):
            t = np.array(t,dtype=float)
    return t

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unnormalize(x,y,alpha,Reynolds,ncrit,Cl,Cd,Cdp,Cm,Cp):    
    with open('min_max_scaler.pickle','rb') as f:
        scalers = pickle.load(f)
    x = scalers['x'].inverse_transform(convert_to_ndarray(x).reshape(-1,1))
    y = scalers['y'].inverse_transform(convert_to_ndarray(y).reshape(-1,1))
    alpha = scalers['alpha'].inverse_transform(convert_to_ndarray(alpha).reshape(-1,1))
    Reynolds = scalers['Reynolds'].inverse_transform(convert_to_ndarray(Reynolds).reshape(-1,1))
    ncrit = scalers['ncrit'].inverse_transform(convert_to_ndarray(ncrit).reshape(-1,1))
    Cl = scalers['Cl'].inverse_transform(convert_to_ndarray(Cl).reshape(-1,1))    
    Cd = scalers['Cd'].inverse_transform(convert_to_ndarray(Cd).reshape(-1,1))
    Cdp = scalers['Cdp'].inverse_transform(convert_to_ndarray(Cdp).reshape(-1,1))
    Cm = scalers['Cm'].inverse_transform(convert_to_ndarray(Cm).reshape(-1,1))
    Cp = scalers['Cp'].inverse_transform(convert_to_ndarray(Cp).reshape(-1,1))
    return x,y,alpha,Reynolds,ncrit,Cl,Cd,Cdp,Cm,Cp

def normalize(x,y,alpha,Reynolds,ncrit,Cl,Cd,Cdp,Cm,Cp):
    with open('min_max_scaler.pickle','rb') as f:
        scalers = pickle.load(f)
    
    x = convert_to_ndarray(x).reshape(-1,1) # scalers['x'].transform(convert_to_ndarray(x).reshape(-1,1))
    y = scalers['y'].transform(convert_to_ndarray(y).reshape(-1,1))    
    alpha = scalers['alpha'].transform(convert_to_ndarray(alpha).reshape(-1,1))
    Reynolds = scalers['Reynolds'].transform(convert_to_ndarray(Reynolds).reshape(-1,1))
    ncrit = scalers['Ncrit'].transform(convert_to_ndarray(ncrit).reshape(-1,1))
    Cl = scalers['Cl'].transform(convert_to_ndarray(Cl).reshape(-1,1))   
    Cd = scalers['Cd'].transform(convert_to_ndarray(Cd).reshape(-1,1))
    Cdp = scalers['Cdp'].transform(convert_to_ndarray(Cdp).reshape(-1,1))
    Cm = scalers['Cm'].transform(convert_to_ndarray(Cm).reshape(-1,1))
    Cp = scalers['Cp'].transform(convert_to_ndarray(Cp).reshape(-1,1))
    return x,y,alpha,Reynolds,ncrit,Cl,Cd,Cdp,Cm,Cp

def csapi(x,y,xx):
    """
        functions similarly to matlab's cubic spline interpolator
    """
    cs = CubicSpline(x,y)
    return cs(xx)

def pchip(x,y,xx):
    pch = PchipInterpolator(x,y)
    return pch(xx)

def save_checkpoint(epoch, model, optimizer,filename):
    """
    Save model checkpoint.    
    @param epoch epoch number
    @param model model
    @param optimizer optimizer
    """
    
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer': optimizer}    
    torch.save(state, filename)

def load_checkpoint(filename):
    '''
    Loads model checkpoint.
    @param filename name of file to load
    '''
        
    state = torch.load(filename)

    return state['epoch'], state['model_state_dict'],state['optimizer']


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_edge_adjacency(npoints:int):
    """This describes how the points are connected with each other

        example: 
                [[0, 1],
                [1, 2],
                [3, 4],
                [4, 0]]
            This says point 0 is connected to 1. 1 is connected to 2 and eventually 4 is connected to 0.
            This edge definition reflects the connectivity of an airfoil geometry. 

    Args:
        npoints (int): Number of points in an airfoil 

    Returns:
        List[(int,int)]: List of point connectivities
        
    """
    edges = list()
    for i in range(1,npoints):
        edges.append([i-1,i])
    edges.append([len(edges),0])
    return edges