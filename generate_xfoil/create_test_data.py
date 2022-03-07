from sklearn import preprocessing
import pickle, json, copy
import numpy as np
import naca4 as naca
from xfoil import xfdata, run_xfoil
import sys
sys.path.insert(0,'../')
from Airfoil import Airfoil
from libs.pspline import pspline
import torch
import os.path as osp
import pandas as pd 
from libs.utils import create_edge_adjacency, normalize, csapi

def run_test_case():
    '''
        runs a single test case
    '''
    naca_foil = '2315'
    if (not osp.exists('{0}.pickle'.format(naca_foil))):
        # Generate the Airfoil 
        n = 90
        [x,y] = naca.naca4(naca_foil,n)
        x = np.array(x)
        y = np.array(y)
        xss = np.append(x[0:n],0)
        yss = np.append(y[0:n],0)

        xps = x[n+1:]
        yps = y[n+1:]
        Reynolds = 50000
        ncrit = 5
        # Save the airfoil and conditions as an xfoil object
        # Run xfoil
        df_list = list()
        for alpha in range(-5,20):
            airfoil = xfdata(airfoil_name=naca_foil,num_nodes=len(xss),
                save_airfoil_filename='saveAF.txt',save_polar_filename='polar_{0}_{1}.txt'.format(Reynolds,ncrit),
                root_folder='xfoil_runs',xfoil_input='xfoil_input.txt',ncrit=ncrit,Reynolds=Reynolds,Mach=0,
                xss=xss,yss=yss,xps=xps,yps=yps,
                alpha_min=alpha,alpha_max=alpha,alpha_step=1)
            df = run_xfoil(airfoil,'xfoil/xfoil.exe',return_cp=True,return_polars=True,check_convergence=True)
            df_list.append(df)
        df = pd.concat(df_list)        
        df.to_pickle('{0}.pickle'.format(airfoil.airfoil_name))
    
    # Load the dataframe
    df = pd.read_pickle('{0}.pickle'.format(naca_foil))
    df = df.dropna()
    # Extract the normalized airfoil
    airfoil_list = list()
    for _ ,row in df.iterrows():
        x_cp_ss = list(); x_cp_ps = list()
        Cp_ss = list(); Cp_ps = list()    
        # * Interpolate XCp and Cp
        for i in range(len(row['x_cp'])):
            if i == 0:
                x_cp_ss.append(row['x_cp'][i])
                Cp_ss.append(row['Cp'][i])
            elif (i>0 and row['x_cp'][i]<x_cp_ss[-1]):
                x_cp_ss.append(row['x_cp'][i])
                Cp_ss.append(row['Cp'][i])
            else:
                break
            
        x_cp_ps = [row['x_cp'][i] for i in range(len(x_cp_ss),len(row['x_cp']))]
        Cp_ps = [row['Cp'][i] for i in range(len(x_cp_ss),len(row['x_cp']))]
        
        x_cp_ss = np.flip(np.array(x_cp_ss))
        Cp_ss = np.flip(np.array(Cp_ss))

        x_cp_ps = np.array(x_cp_ps)
        Cp_ps = np.array(Cp_ps)

        Cp_ss = csapi(x_cp_ss,Cp_ss,df['xss'].iloc[0]) # from 0 to 1
        Cp_ps = csapi(x_cp_ps,Cp_ps,df['xps'].iloc[0])

        Cp = np.concatenate(( Cp_ss, np.flip(Cp_ps)[1:-1] ))
        x = np.concatenate(( row['xss'], np.flip(row['xps'])[1:-1] ))
        y = np.concatenate(( row['yss'], np.flip(row['yps'])[1:-1] ))

        [x,y,alpha,Reynolds,ncrit,Cl,Cd,Cdp,Cm,Cp] = normalize(x,y,row['alpha'],row['Reynolds'],row['Ncrit'],
                                    row['Cl'],row['Cd'],row['Cdp'],row['Cm'],Cp)
        node_labels = torch.as_tensor(Cp,dtype=torch.float32)
        edge_index = torch.as_tensor(create_edge_adjacency(x.shape[0]), dtype=torch.long).t().contiguous()
        ones = np.ones(x.shape)

        data_x = torch.as_tensor(np.hstack([y]), dtype=torch.float32)
        conditions = torch.as_tensor([alpha[0][0], Reynolds[0][0], ncrit[0][0]],dtype=torch.float32)
        pos = torch.as_tensor(np.hstack([x, y]), dtype=torch.float32)
        data_y = torch.as_tensor(np.hstack([Cl[0], Cd[0], Cdp[0], Cm[0] ]),dtype=torch.float32)
        edge_attr = torch.ones((edge_index.shape[1],1),dtype=torch.float32)

        airfoil_list.append(Airfoil(x=data_x,edge_index=edge_index, pos=pos, conditions=conditions,y=data_y,node_labels=node_labels))
    torch.save(airfoil_list,'naca_'+naca_foil+'.pt')    
    

if __name__ == "__main__":
    run_test_case()
    
    