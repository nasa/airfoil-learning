''''
    Reads through all the Json files and resizes the Cp. Writes the data back to a separate json_cp_resize folder
'''
import os, glob
import os.path as osp
from libs.utils import pchip
import json 
from tqdm import trange 
import numpy as np 
import copy

new_save_dir = 'json_cp_resize'
os.makedirs(new_save_dir,exist_ok=True)

cp_points = 50

x_cp = np.linspace(0,1,cp_points)
data_files = glob.glob(osp.join('json','*.json'))
pbar = trange(len(data_files),desc='Processing')
for i in pbar:
    filename = data_files[i]
    with open(filename,'r') as f:
        airfoil = json.load(f)
        new_airfoil = copy.deepcopy(airfoil)

        xss = airfoil['xss']        
        yss = airfoil['yss']

        xps = airfoil['xps']
        yps = airfoil['yps']

        for p in range(len(airfoil['polars'])):            
            polar = airfoil['polars'][p]

            Cp_ss = np.array(polar['Cp_ss'])
            Cp_ps = np.array(polar['Cp_ps'])

            # Resize cp
            Cp_ss = pchip(xss,Cp_ss,x_cp) # from 0 to 1
            Cp_ps = pchip(xps,Cp_ps,x_cp) # use xss here because we need same length for cp
            
            new_airfoil['polars'][p]['Cp_ss'] = Cp_ss.tolist()
            new_airfoil['polars'][p]['Cp_ps'] = Cp_ps.tolist()

    with open(osp.join(new_save_dir,osp.basename(filename)),'w') as f2:
        json.dump(new_airfoil,f2)
