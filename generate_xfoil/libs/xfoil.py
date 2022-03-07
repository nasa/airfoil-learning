import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from .xfoil_lib import read_polar, read_cp
import pandas as pd
import math
import subprocess
import time, platform, signal
import ctypes

@dataclass
class xfdata:
    '''
        Input structure into XFOIL
    '''    
    airfoil_name:str 
    alpha_min:float
    alpha_max:float
    alpha_step:float
    ncrit:float
    Reynolds:float
    Mach:float
    num_nodes:int  
    save_airfoil_filename:str 
    save_polar_filename:str  # save results filename
    root_folder:str
    xfoil_input:str
    xss: np.ndarray
    yss: np.ndarray
    xps: np.ndarray
    yps: np.ndarray

def create_psav(xfoil_data:xfdata):
    """Save the airfoil in plain coordinate format for xfoil to read 

    Args:
        xfoil_data (xfdata): class that contains everything needed to call xfoil 
    """
    with open(osp.join(xfoil_data.root_folder, xfoil_data.save_airfoil_filename),'w') as f:
        for i in range(len(xfoil_data.xss)):
            f.write(' {0:0.7f} {1:0.7f}\n'.format(xfoil_data.xss[i],xfoil_data.yss[i]))
        for i in range(len(xfoil_data.xps)):
            f.write(' {0:0.7f} {1:0.7f}\n'.format(xfoil_data.xps[i],xfoil_data.yps[i]))

def run_xfoil(xfoil_data:xfdata,exe_path:str,return_cp=True,return_polars=True, check_convergence=True):
    """Runs xfoil simulation. 

    Args:
        xfoil_data (xfdata): Class which represents the inputs to pass to xfoil 
        exe_path (str): path to xfoil executable 
        return_cp (bool, optional): Returns the cp value. Defaults to True.
        return_polars (bool, optional): Returns only Cl,Cd,Cdp,Cm. Defaults to True.
        check_convergence (bool, optional): Checks to see if xfoil simulation converged. Defaults to True.

    Returns:
        pd.dataframe or None: If simulation did not converge then None is returned otherwise a dataframe with the results is returned

    """
    for item in os.listdir(xfoil_data.root_folder):
        if item.endswith(".txt") or item.endswith(".cp"):
            os.remove(os.path.join(xfoil_data.root_folder, item))
    
    create_psav(xfoil_data) # Create the airfoil data

    def get_cp(aoa):
        # Get the Cp
        with open(osp.join(xfoil_data.root_folder,xfoil_data.xfoil_input),"w") as fid:
            # Add Operation        
            fid.write('LOAD {0}\n{1}\n'.format(str(osp.join(xfoil_data.root_folder,xfoil_data.save_airfoil_filename)),xfoil_data.airfoil_name))
            fid.write('PPAR\nN\n{0}\n\n\n'.format(xfoil_data.num_nodes))
            fid.write('GDES\nCADD\n\n\n\n\n'.format(xfoil_data.num_nodes))

            fid.write("OPER\n")
            fid.write('VPAR\n')
            fid.write('N \n{0}\n'.format(xfoil_data.ncrit))
            fid.write('XTR \n{0}\n{1}\n\n'.format(1.0,1.0))

            fid.write('VISC {0}\n'.format(xfoil_data.Reynolds))
            fid.write('MACH {0}\n'.format(xfoil_data.Mach))
            fid.write('ITER {0}\n'.format(100))
            fid.write('ALFA 0.0\n')
            fid.write('INIT\n')
            fid.write('PACC \n{0}\ntemp.dump\n'.format('temp_polar.txt'))

            fid.write('ALFA {0}\n'.format(aoa))    
            fid.write("CPWR \n {0}.cp\n\n\n\n".format(osp.join(xfoil_data.root_folder,'Cp_'+ str(aoa))))
            fid.write("QUIT" + "\n")
        
        os.system("timeout 0.5s {0} < {1}".format(exe_path,osp.join(xfoil_data.root_folder,xfoil_data.xfoil_input)))
        return osp.join(xfoil_data.root_folder,'Cp_'+str(aoa)+'.cp')   

    def get_polars():
        # Get the Polar (Cl vs angle of attack)
        

        with open(osp.join(xfoil_data.root_folder,xfoil_data.xfoil_input),"w") as fid:
            # Add Operation        
            fid.write('LOAD {0}\n{1}\n'.format(str(osp.join(xfoil_data.root_folder,xfoil_data.save_airfoil_filename)),xfoil_data.save_airfoil_filename))
            fid.write('PPAR\nN\n{0}\n\n\n'.format(xfoil_data.num_nodes))
            fid.write('GDES\nCADD\n\n\n\n\n'.format(xfoil_data.num_nodes))

            fid.write("OPER\n")
            fid.write('VPAR\n')
            fid.write('N \n{0}\n'.format(xfoil_data.ncrit))
            fid.write('XTR \n{0}\n{1}\n\n'.format(1.0,1.0))

            fid.write('VISC {0}\n'.format(xfoil_data.Reynolds))
            fid.write('MACH {0}\n'.format(xfoil_data.Mach))        
            fid.write('ITER {0}\n'.format(1000))
            fid.write('ASEQ\n')
            fid.write('{0}\n'.format(xfoil_data.alpha_min))
            fid.write('{0}\n'.format(xfoil_data.alpha_max))
            fid.write('{0}\n'.format(xfoil_data.alpha_step))

            fid.write('INIT\n')
            fid.write('PACC\n{0}\ntemp.dump\n'.format(osp.join(xfoil_data.root_folder, xfoil_data.save_polar_filename)))

            fid.write('ASEQ \n{0}\n{1}\n{2}\n'.format(xfoil_data.alpha_min,xfoil_data.alpha_max,xfoil_data.alpha_step))
            # fid.write("CPWR \n polar_temp.cp\n\n\n\n")
            fid.write("QUIT" + "\n")
            # os.system("xfoil < {0}".format(xfoil_data.xfoil_input))
            
        os.system("timeout 1s {0} < {1}".format(exe_path,osp.join(xfoil_data.root_folder,xfoil_data.xfoil_input)))

        
    if osp.exists('temp.dump'): # delete the dump file
        os.remove('temp.dump')
    if osp.exists('temp_polar.txt'): # delete the temp polar file
        os.remove('temp_polar.txt')

    if (return_cp and return_polars):
        get_polars()
        df_polar = read_polar(osp.join(xfoil_data.root_folder,xfoil_data.save_polar_filename))
        df_polar['x_cp'] = math.nan
        df_polar['Cp'] = math.nan
        df_polar['xss'] = None
        df_polar['yss'] = None
        df_polar['xps'] = None
        df_polar['yps'] = None
        df_polar=df_polar.astype(object)
        if (xfoil_data.alpha_min == xfoil_data.alpha_max):
            aoa_cp_file = get_cp(xfoil_data.alpha_min)
            x,cp = read_cp(aoa_cp_file)
            row = df_polar.index[df_polar['alpha'] == xfoil_data.alpha_min]
            if len(row)>0:
                row = row.values[0]
                df_polar.at[row,'xss'] = np.flip(xfoil_data.xss).tolist()
                df_polar.at[row,'yss'] = np.flip(xfoil_data.yss).tolist()
                df_polar.at[row,'xps'] = np.flip(xfoil_data.xps)[1:].tolist() # TODO: Need to double check this for other airfoils, works with test naca foil
                df_polar.at[row,'yps'] = np.flip(xfoil_data.yps)[1:].tolist()
                if (x is not None and cp is not None):
                    df_polar.at[row,'x_cp'] = x.tolist()
                    df_polar.at[row,'Cp'] = cp.tolist()
        else:
            aoa_list = np.arange(xfoil_data.alpha_min, xfoil_data.alpha_max,xfoil_data.alpha_step)
            for aoa in aoa_list:
                aoa_cp_file = get_cp(aoa)
                x,cp = read_cp(aoa_cp_file)
                row = df_polar.index[df_polar['alpha'] == aoa]
                if len(row)>0:
                    row = row.values[0]
                    df_polar.at[row,'xss'] = np.flip(xfoil_data.xss).tolist()
                    df_polar.at[row,'yss'] = np.flip(xfoil_data.yss).tolist()
                    df_polar.at[row,'xps'] = np.flip(xfoil_data.xps)[1:].tolist() # TODO: Need to double check this for other airfoils, works with test naca foil
                    df_polar.at[row,'yps'] = np.flip(xfoil_data.yps)[1:].tolist()
                    if (x is not None and cp is not None):
                        df_polar.at[row,'x_cp'] = x.tolist()
                        df_polar.at[row,'Cp'] = cp.tolist()
    elif (return_cp): # This is in case you already have the polar file and just want the Cp for that one polar
        aoa_cp_file = get_cp(xfoil_data.alpha_min)
        x,cp = read_cp(aoa_cp_file)
        # Also read temp polar to see if it's converged
        df_temp = read_polar('temp_polar.txt')
        if check_convergence:             
            if not df_temp.empty: # didn't converge            
                if x is not None and cp is not None:
                    data = {'x':x, 'Cp':cp}
                    df_polar = pd.DataFrame(data)
                else:
                    df_polar= None
            else:
                df_polar = None # didn't converge
        else:
            data = {'x':x, 'Cp':cp}
    try:
        if osp.exists('temp.dump'): # delete the dump file
            os.remove('temp.dump')
        if osp.exists('temp_polar.txt'): # delete the temp polar file
            os.remove('temp_polar.txt')
    except:
        print("cannot remove dump file ... skipping")
    return df_polar 


def check_PID_running(pid):
    """
        Checks if a pid is still running (UNIX works, windows we'll see)
        Inputs:
            pid - process id
        returns:
            True if running, False if not
    """
    if (platform.system() == 'Linux'):
        try:
            os.kill(pid, 0)
        except OSError:
            return False 
        else:
            return True
    elif (platform.system() == 'Windows'):
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x100000

        process = kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
        if process != 0:
            kernel32.CloseHandle(process)
            return True
        else:
            return False