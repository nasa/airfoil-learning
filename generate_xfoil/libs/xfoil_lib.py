import os.path as osp
import glob
import pandas as pd
import sys
import numpy as np

def read_polar(polar_file):
    airfoil_data = list()
    try:
        # Read Mach, Reynolds Number, Ncrit
        if 'polar_' in polar_file:
            data = osp.basename(polar_file).split('polar_')[1].replace('.txt','').split('_')
            Reynolds = float(data[0])
            Ncrit = float(data[1])        
        else:
            Reynolds = 99999999
            Ncrit = 99999999
        
        with open(polar_file,'r') as fp:        
            [fp.readline() for i in range(12)] # Skip 11 lines
            for _, line in enumerate(fp):
                line = line.strip()
                data = [float(d) for d in line.split(' ') if d != '']
                data.append(Reynolds)
                data.append(Ncrit)
                airfoil_data.append(data)
        if len(airfoil_data)!=0 and len(airfoil_data[-1]) == 2: # if you have an empty line then it's just reynolds and ncrit being added
            airfoil_data.pop()
    except:
        print("something terrible has happened")
        airfoil_data = list()
        

    if airfoil_data:
        if len(airfoil_data[0])==11:
            columns = ['alpha','Cl','Cd','Cdp','Cm','ss_xtr','ps_xtr','Top_Itr','Bot_Itr','Reynolds','Ncrit']
        else:
            columns = ['alpha','Cl','Cd','Cdp','Cm','ss_xtr','ps_xtr','Reynolds','Ncrit']
        df = pd.DataFrame(data=airfoil_data,columns=columns)
    else:
        df= pd.DataFrame()
    return df

def read_cp(cp_file):
    # Load the data from the text file
    try:
        dataBuffer = np.loadtxt(cp_file, skiprows=1)
        # Extract data from the loaded dataBuffer array
        X_0  = dataBuffer[:,0]
        # Y_0  = dataBuffer[:,1]
        Cp_0 = dataBuffer[:,1]
    except:
        X_0 = None
        Cp_0 = None
    return X_0, Cp_0
