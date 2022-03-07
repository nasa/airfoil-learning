'''
    Step 2
    This file reads over all the scrapped airfoil data in the "scrape" folder and runs xfoil
    creates a file called

    This file needs to be run in WSL for it to work
'''
import glob

from pandas.core.reshape.concat import concat
from libs.utils import pchip
import os, copy
import os.path as osp
from tqdm import trange
import pandas as pd
import logging
import numpy as np
import json
import time 
from libs.xfoil_lib import read_polar
from libs.xfoil import xfdata, run_xfoil


def coordinate_read(coord_file,npoints=100):
    '''
        Reads the coordinates of the airfoil
        @params coord_file - file to read
        @params npoints - number of points used to represent suction and pressure side, 0 - use what's in the file
    '''
    x = list(); y = list()
    airfoil_name = osp.basename(coord_file).replace('.txt', ' ').strip()

    with open(coord_file,'r') as fp:
        for cnt, line in enumerate(fp):
            if cnt>2:
                data = line.strip().split(' ')
                data = [i for i in data if i] 
                if (len(data)==2):
                    x.append(float(data[0]))
                    y.append(float(data[1]))
                else:
                    xss = copy.deepcopy(x); yss = copy.deepcopy(y)
                    x.clear()
                    y.clear()
        xps = copy.deepcopy(x)
        yps = copy.deepcopy(y)

        xps2 = list()
        yps2 = list()
        xss2 = list()
        yss2 = list()
        for i in range(len(xps)):               # This part makes sure x is strictly increasing sequence for Pressure side
            if (xps[i] not in xps2):
                if i>0 and xps[i]>xps2[-1]:
                    xps2.append(xps[i])
                    yps2.append(yps[i])
                elif i==0:
                    xps2.append(xps[i])
                    yps2.append(yps[i])
        for i in range(len(xss)):               # This part makes sure x is strictly increasing sequence for Suction side
            if (xss[i] not in xss2):
                if i>0 and xss[i]>xss2[-1]:
                    yss2.append(yss[i])
                    xss2.append(xss[i])
                elif i==0:
                    yss2.append(yss[i])
                    xss2.append(xss[i])
        xss = xss2; yss=yss2; xps=xps2; yps=yps2

        if npoints>0:
            x = np.linspace(min(xss),max(xss),npoints)
            yss = pchip(xss,yss,x)
            xss = x
            yps = pchip(xps,yps,x)
            xps = x
            # yss[-1] = yps[-1]

    return xss,yss,xps,yps,airfoil_name

def process_scaped_data(npoints=100,check_convergence=True,save_dir:str='json'):
    """Reads the scraped airfoil .txt file, runs xfoil, save to json
        This is an important step in the data processing. This step normalizes the data so that for each node on suction (ss) and pressure (ps) side you have a Cp value 

    Args:
        npoints (int, optional): number of points on pressure and suction size. You have to normalize the data to a certain number of points. Defaults to 90.
        check_convergence (bool, optional): Checks for convergence when running xfoil. Defaults to True.

    Returns:
        None: Saves the data to json 
    """

    unprocessed_filename = 'data/scraped_airfoils_no_cp.gz'
    # * Step 1 Read through the scape data
    if not osp.exists(unprocessed_filename):
        files = glob.glob('data/scrape/*.txt')
        files.sort()
        airfoil_files = [f for f in files if 'polar' not in f]

        airfoil_data = list()        
        for f in trange(len(airfoil_files),desc='Reading Scraped Airfoils'):
            xss,yss,xps,yps,airfoil_name = coordinate_read(airfoil_files[f])
            # search for all the polars of that airfoil
            polar_list = list()
            polar_files = glob.glob('data/scrape/{0}_polar*'.format(airfoil_name))
            for p in range(len(polar_files)):
                polar_df = read_polar(polar_files[p])
                polar_df['Cp_ss'] = None # Add a column for Cp (will be calculated later)
                polar_df['Cp_ps'] = None # Add a column for Cp (will be calculated later)                            
                polar_list.append(polar_df)
            polar_df = pd.concat(polar_list)
            
            temp = {'name':airfoil_name,'xss':xss,'yss':yss,'xps':xps,'yps':yps,'polars':polar_df}
            airfoil_data.append(temp)
        df = pd.DataFrame(airfoil_data)
        df.to_pickle('data/scraped_airfoils_no_cp.gz')

    # * Step 2 Generate Cp
    df = pd.read_pickle(unprocessed_filename)
    x = np.linspace(0,1,npoints) # ! This fixes the number of points

    os.makedirs('json', exist_ok=True)
    df = df.sort_values('name')
    for p in trange(0, len(df),desc='Calculating Cp for each condition'): # Loop for each Airfoil        
        airfoil = df.iloc[p]        
        if not osp.exists(osp.join(save_dir,airfoil['name']+'.json')):
            logging.debug('Reading Airfoil {0}'.format(airfoil['name']))

            xss = np.array(airfoil['xss']); yss = np.array(airfoil['yss']); xps = np.array(airfoil['xps']); yps = np.array(airfoil['yps'])        
            # Interpolate airfoil to 80 points ss and ps
            yss = pchip(xss,yss,x)
            yps = pchip(xps,yps,x)
            xss = x;  xps = x
            
            airfoil_dict = {'name':airfoil['name'],'xss':xss.tolist(),'yss':yss.tolist(),'xps':xps.tolist(),'yps':yps.tolist(),'polars':[]}
            
            # Airfoil data: x must go from 1 to 0 to 1 
            xss = np.flip(xss)[0:-1]
            yss = np.flip(yss)[0:-1]

            polars = airfoil['polars']
            for q in range(len(polars)): # Iterate through all test conditions                                
                row = polars.iloc[q]
                polar_dict = {'alpha':row.alpha,'Cl':row.Cl, 'Cd':row.Cd, 'Cdp':row.Cdp,'Cm':row.Cm,'Re':row.Reynolds,'Ncrit':row.Ncrit,'alpha_min':row.alpha,'alpha_max':row.alpha,'alpha_step':1,'Cp_ss':[],'Cp_ps':[]}

                xfoil_data = xfdata(airfoil_name=airfoil['name'],num_nodes=len(xss),
                    save_airfoil_filename='saveAF.txt',save_polar_filename='polar_{0}_{1}.txt'.format(row.Reynolds,row.Ncrit),
                    root_folder='xfoil_runs',xfoil_input='xfoil_input.txt',ncrit=row.Ncrit,Reynolds=row.Reynolds,Mach=0,
                    xss=xss,yss=yss,xps=xps,yps=yps,
                    alpha_min=row.alpha,alpha_max=row.alpha,alpha_step=1)

                time.sleep(0.35) # need this for stability
                xfoil_polar = run_xfoil(xfoil_data,'xfoil',return_cp=True,return_polars=False,check_convergence=check_convergence) # Run xfoil           
                if xfoil_polar is not None:
                    Cp_ss = list(); Cp_ps = list()
                    x_cp_ss = list(); x_cp_ps = list()
                    x_cp = xfoil_polar['x']
                    cp = xfoil_polar['Cp']
                    # Interpolate XCp and Cp
                    for i in range(len(x_cp)):
                        if i == 0:
                            x_cp_ss.append(x_cp[i])
                            Cp_ss.append(cp[i])
                        elif (i>0 and x_cp[i]<x_cp_ss[-1]):
                            x_cp_ss.append(x_cp[i])
                            Cp_ss.append(cp[i])
                        else:
                            break
                        
                    x_cp_ps = [x_cp[i] for i in range(len(x_cp_ss),len(x_cp))]
                    Cp_ps = [cp[i] for i in range(len(x_cp_ss),len(x_cp))]
                    
                    

                    x_cp_ss = np.flip(np.array(x_cp_ss))
                    Cp_ss = np.flip(np.array(Cp_ss))

                    x_cp_ps = np.array(x_cp_ps)
                    Cp_ps = np.array(Cp_ps)

                    Cp_ss = pchip(x_cp_ss,Cp_ss,xps) # from 0 to 1
                    Cp_ps = pchip(x_cp_ps,Cp_ps,xps) # use xss here because we need same length for cp
                    polar_dict['Cp_ss'] = Cp_ss.tolist()   
                    polar_dict['Cp_ps'] = Cp_ps.tolist()
                    
                    airfoil_dict['polars'].append(polar_dict)
                    

            with open(osp.join(save_dir,airfoil['name'] + '.json'),'w') as f:
                json.dump(airfoil_dict, f,indent=4, sort_keys=True)


# def clean_scraped_data(scraped_filename):
#     '''
#         Remove None and NaN's
#     '''
#     df = pd.read_pickle(scraped_filename)
#     df.dropna()
#     droprows = list()
#     for a in range(len(df)):
#         airfoil = df.iloc[a]
#         polars = airfoil['polars']

#         polars = polars.dropna()      
#         df.at[a,'polars'] = polars

        
#         # for p in range(len(polars)): # TODO if you recreate the data, this step may not be needed 
#         #     if (len(polars.iloc[p]['Cp_ps']) > len(polars.iloc[p]['Cp_ss']) ):
#         #         Cp_ps = np.flip(polars.iloc[p]['Cp_ps'])  # unflip cp_ps
#         #         Cp_ss = np.concatenate([polars.iloc[p]['Cp_ss'],[Cp_ps[-1]]]) # Make it the same length
#         #         polars.iat[p,9] = Cp_ss
#         #         polars.iat[p,10] = Cp_ps
#         # df.at[a,'polars'] = polars
#         if polars.empty:            # polar is empty remove the row
#             droprows.append(a)
#     df = df.drop(labels=droprows)
#     df.to_pickle(osp.splitext(scraped_filename)[0] + "_clean.gz")
#     return df

if __name__ == "__main__":
    processed_filename = process_scaped_data(npoints=100,check_convergence=True)
    # clean_scraped_data(processed_filename)