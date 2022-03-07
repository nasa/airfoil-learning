'''
    This version changes
    - Node features are the coordinates of the node scaled. edge_attributes are unscaled in order to fully capture the spline
    - In addition to all the changes below.
    Version 02 changes
    - Edge Attribute contains the x and y points of the edge start node as oppose to 1's. 
        SplineConv uses the edge_attribute feature to compute the spline basis function
    - Simplified the export code (create_torch_geometric_train_test_data), less parameters now
    - Cl,Cd,Cm,Cdp are scaled by their min and max. Cp is scaled as a group
'''
import os.path as osp
import glob, copy, random, json, os
import pandas as pd
import numpy as np
import math
import sys
from Airfoil import Airfoil
import pspline
from xfoil_lib import read_polar
from xfoil import xfdata, run_xfoil
from tqdm import trange
from sklearn import preprocessing
from utils import normalize, csapi
import logging, pickle
import torch
from torch.utils.data import random_split
logging.basicConfig(filename='debug.log',level=logging.DEBUG)


def coordinate_read(coord_file,npoints=50):
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
        xps = x
        yps = y
        xps2 = list()
        yps2 = list()
        xss2 = list()
        yss2 = list()
        for i in range(len(xps)):
            if (xps[i] not in xps2):
                if i>0 and xps[i]>xps2[-1]:
                    xps2.append(xps[i])
                    yps2.append(yps[i])
                elif i==0:
                    xps2.append(xps[i])
                    yps2.append(yps[i])
        for i in range(len(xss)):            
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
            yss = csapi(xss,yss,x)
            xss = x
            yps = csapi(xps,yps,x)
            xps = x
            # yss[-1] = yps[-1]
    return xss,yss,xps,yps,airfoil_name

def process_scaped_data(npoints=90,check_convergence=True):
    # * Step 1 Read through the scape data
    if not osp.exists('scraped_airfoils_no_cp.gz'):
        files = glob.glob('scrape/*.txt')
        files.sort()
        airfoil_files = [f for f in files if 'polar' not in f]

        airfoil_data = list()        
        for f in trange(len(airfoil_files),desc='Reading Scraped Airfoils'):
            xss,yss,xps,yps,airfoil_name = coordinate_read(airfoil_files[f])
            # search for all the polars of that airfoil
            polar_list = list()
            polar_files = glob.glob('scrape/{0}_polar*'.format(airfoil_name))
            for p in range(len(polar_files)):
                polar_df = read_polar(polar_files[p])
                polar_df['Cp_ss'] = None # Add a column for Cp (will be calculated later)
                polar_df['Cp_ps'] = None # Add a column for Cp (will be calculated later)                            
                polar_list.append(polar_df)
            polar_df = pd.concat(polar_list)
            
            temp = {'name':airfoil_name,'xss':xss,'yss':yss,'xps':xps,'yps':yps,'polars':polar_df}
            airfoil_data.append(temp)
        df = pd.DataFrame(airfoil_data)
        df.to_pickle('scraped_airfoils_no_cp.gz')

    # * Step 2 Generate Cp
    df = pd.read_pickle('scraped_airfoils_no_cp.gz')
    x = np.linspace(0,1,npoints) # ! This fixes the number of points

    os.makedirs('airfoil_cp_temp', exist_ok=True)     
    airfoil_list = glob.glob('airfoil_cp_temp/*.gz')
    if (len(airfoil_list)==1):
        start_index = osp.basename(airfoil_list[0]).replace('.gz','')
    else:
        start_index = 0

    for p in trange(start_index, len(df),desc='Calculating Cp for each condition'): # Loop for each Airfoil
        airfoil = df.iloc[p]
        logging.debug('Reading Airfoil {0}'.format(airfoil['name']))

        xss = np.array(airfoil['xss']); yss = np.array(airfoil['yss']); xps = np.array(airfoil['xps']); yps = np.array(airfoil['yps'])        
        # Interpolate airfoil to 80 points ss and ps
        yss = csapi(xss,yss,x)
        yps = csapi(xps,yps,x)
        xss = x;  xps = x
        
        airfoil['xss'] = xss
        airfoil['xps'] = xps
        airfoil['yps'] = yps
        airfoil['yss'] = yss
        # Airfoil data: x must go from 1 to 0 to 1 
        xss = np.flip(xss)[0:-1]
        yss = np.flip(yss)[0:-1]

        polars = airfoil['polars']
        for q in range(len(polars)): # Iterate through all test conditions
            row = polars.iloc[q]
            xfoil_data = xfdata(airfoil_name=airfoil['name'],num_nodes=len(xss),
                save_airfoil_filename='saveAF.txt',save_polar_filename='polar_{0}_{1}.txt'.format(row.Reynolds,row.Ncrit),
                root_folder='xfoil_runs',xfoil_input='xfoil_input.txt',ncrit=row.Ncrit,Reynolds=row.Reynolds,Mach=0,
                xss=xss,yss=yss,xps=xps,yps=yps,
                alpha_min=row.alpha,alpha_max=row.alpha,alpha_step=1)

            xfoil_polar = run_xfoil(xfoil_data,'xfoil/xfoil.exe',return_cp=True,return_polars=False,check_convergence=check_convergence) # Run xfoil           
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

                Cp_ss = csapi(x_cp_ss,Cp_ss,airfoil['xss']) # from 0 to 1
                Cp_ps = csapi(x_cp_ps,Cp_ps,xps)
                polars.iat[q,9] = Cp_ss.tolist() # 9 - should be Cp_ss
                polars.iat[q,10] = Cp_ps.tolist() 
        df.at[p,'polars'] = polars        
        df.to_pickle('airfoil_cp_temp/{0:04d}.gz'.format(p))
    df.to_pickle('scraped_airfoils_cp.gz')

    # * Step 2 Generate Cp
    df = pd.read_pickle('scraped_airfoils_no_cp.gz')
    x = np.linspace(0,1,npoints) # ! This fixes the number of points

    os.makedirs('airfoil_cp_temp', exist_ok=True)     
    airfoil_list = glob.glob('airfoil_cp_temp/*.gz')
    for p in trange(len(airfoil_list), len(df),desc='Calculating Cp for each condition'): # Loop for each Airfoil
        airfoil = df.iloc[p]
        logging.debug('Reading Airfoil {0}'.format(airfoil['name']))

        xss = np.array(airfoil['xss']); yss = np.array(airfoil['yss']); xps = np.array(airfoil['xps']); yps = np.array(airfoil['yps'])        
        # Interpolate airfoil to 80 points ss and ps
        yss = csapi(xss,yss,x)
        yps = csapi(xps,yps,x)
        xss = x;  xps = x
        
        airfoil['xss'] = xss
        airfoil['xps'] = xps
        airfoil['yps'] = yps
        airfoil['yss'] = yss
        # Airfoil data: x must go from 1 to 0 to 1 
        xss = np.flip(xss)[0:-1]
        yss = np.flip(yss)[0:-1]

        polars = airfoil['polars']
        for q in range(len(polars)): # Iterate through all test conditions
            row = polars.iloc[q]
            xfoil_data = xfdata(airfoil_name=airfoil['name'],num_nodes=len(xss),
                save_airfoil_filename='saveAF.txt',save_polar_filename='polar_{0}_{1}.txt'.format(row.Reynolds,row.Ncrit),
                root_folder='xfoil_runs',xfoil_input='xfoil_input.txt',ncrit=row.Ncrit,Reynolds=row.Reynolds,Mach=0,
                xss=xss,yss=yss,xps=xps,yps=yps,
                alpha_min=row.alpha,alpha_max=row.alpha,alpha_step=1)

            xfoil_polar = run_xfoil(xfoil_data,'xfoil/xfoil.exe',return_cp=True,return_polars=False,check_convergence=check_convergence) # Run xfoil           
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

                Cp_ss = csapi(x_cp_ss,Cp_ss,airfoil['xss']) # from 0 to 1
                Cp_ps = csapi(x_cp_ps,Cp_ps,xps)
                polars.iat[q,9] = Cp_ss.tolist() # 9 - should be Cp_ss
                polars.iat[q,10] = Cp_ps.tolist() 
        df.at[p,'polars'] = polars
        airfoil = df.iloc[p]
        pd.DataFrame(airfoil).to_pickle('airfoil_cp_temp/{0:04d}.gz'.format(p))
    # Concatenate all files in airfoil_cp_temp 
    airfoil_list = glob.glob('airfoil_cp_temp/*.gz')    
    df = pd.concat([pd.read_pickle(airfoil_list[i]) for i in range(len(airfoil_list))])
    df.to_pickle('scraped_airfoils_cp.gz')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_edge_adjacency(npoints):
    edges = list()
    for i in range(1,npoints):
        edges.append([i-1,i])
    edges.append([len(edges),0])
    return edges

def clean_scraped_data(scraped_filename):
    '''
        Remove None and NaN's
    '''
    df = pd.read_pickle(scraped_filename)
    df.dropna()
    droprows = list()
    for a in range(len(df)):
        airfoil = df.iloc[a]
        polars = airfoil['polars']

        polars = polars.dropna()      
        df.at[a,'polars'] = polars

        
        # for p in range(len(polars)): # TODO if you recreate the data, this step may not be needed 
        #     if (len(polars.iloc[p]['Cp_ps']) > len(polars.iloc[p]['Cp_ss']) ):
        #         Cp_ps = np.flip(polars.iloc[p]['Cp_ps'])  # unflip cp_ps
        #         Cp_ss = np.concatenate([polars.iloc[p]['Cp_ss'],[Cp_ps[-1]]]) # Make it the same length
        #         polars.iat[p,9] = Cp_ss
        #         polars.iat[p,10] = Cp_ps
        # df.at[a,'polars'] = polars
        if polars.empty:            # polar is empty remove the row
            droprows.append(a)
    df = df.drop(labels=droprows)
    df.to_pickle(osp.splitext(scraped_filename)[0] + "_clean.gz")
    return df

def CreateMinMaxScalers(scraped_filename):
    columns = ['xss','yss','xps','yps']
    polar_columns = ['alpha','Cl','Cd','Cdp','Cm','Reynolds','Ncrit']
    polar_columns_lists = ['Cp_ss','Cp_ps']
    df = pd.read_pickle(scraped_filename)

    # Lets find a happy medium to normalize the data by selecting random data 
    min_max = dict()        
    min_max_y = list()
    min_max_cp = list()
    for a in trange(len(df)): # loop for all the airfoils
        airfoil = df.iloc[a]
        for col in columns:                                         # min_max for xss,yss,xps,yps
            if col not in min_max.keys():
                min_max[col] = [airfoil[col].min(), airfoil[col].max()]
            else:
                if min(airfoil[col]) < min_max[col][0]:
                    min_max[col][0] = min(airfoil[col])
                if max(airfoil[col]) > min_max[col][1]:
                    min_max[col][1] = max(airfoil[col])
        # Calculate min_max for each y1,y2,y3,..,yn
                    
        polars = df.iloc[a]['polars']
        for col in polar_columns:                                   # min_max for alpha, Cl, Cd, Cdp, Cm, etc
            if col not in min_max.keys():
                min_max[col] = [min(polars[col]), max(polars[col])]
            else:
                if min(polars[col]) < min_max[col][0]:
                    min_max[col][0] = min(polars[col])
                if max(polars[col]) > min_max[col][1]:
                    min_max[col][1] = max(polars[col])
        
        airfoil_y = airfoil['yss'].tolist()
        airfoil_y.extend(np.flip(airfoil['yps'][1:-1]).tolist())
        for i in range(len(airfoil_y)):
            if (len(min_max_y) < len(airfoil_y)):
                min_max_y.append([ airfoil_y[i],airfoil_y[i] ])
            else:
                min_max_y[i] = [ min([ min_max_y[i][0],airfoil_y[i] ]), max([ min_max_y[i][1], airfoil_y[i] ]) ]

        for p in range(len(polars)):                                    
            polar = polars.iloc[p]
            for col in polar_columns_lists:                         # min_max for Cp_ss, Cp_ps
                if col not in min_max.keys():
                    min_max[col] = [min(polar[col]), max(polar[col])]
                else:
                    if min(polar[col]) < min_max[col][0]:
                        min_max[col][0] = min(polar[col])
                    if max(polar[col]) > min_max[col][1]:
                        min_max[col][1] = max(polar[col])
            Cp = polar['Cp_ss']
            Cp_ps = [ polar['Cp_ps'][x] for x in range(1,len(polar['Cp_ps'])-1) ]
            Cp_ps.reverse()
            Cp.extend(Cp_ps)
            for i in range(len(Cp)):
                if (len(min_max_cp) <= len(airfoil_y)):
                    min_max_cp.append([ Cp[i], Cp[i] ])
                else:
                    min_max_cp[i] = [ min([ min_max_cp[i][0],Cp[i] ]), max([ min_max_cp[i][1], Cp[i] ]) ]
                # Calculate min_max for each Cp1,Cp2,Cp_n
        min_max['y'] = [min([ min_max['yss'][0],min_max['yps'][0] ]), max([ min_max['yss'][1],min_max['yps'][1] ])]
        min_max['Cp'] = [min([ min_max['Cp_ss'][0],min_max['Cp_ps'][0] ]), max([ min_max['Cp_ss'][1],min_max['Cp_ps'][1] ])]
        # ! Overiding min_max['Cp']
        min_max['Cp'] = [-2,1.2] # most data is around this
        
        # * Save scalers for y1...yN
        scalers = dict()        
        for k in range(len(min_max_y)):
            scalers[k] = preprocessing.MinMaxScaler(feature_range=(0,1))
            scalers[k].fit(np.array(min_max_y[k]).reshape(-1,1))

        with open('min_max_y.pickle','wb') as f: # min max of individual y and Cp positions
            pickle.dump(scalers,f)

        # * Save scalers for Cp1...CpN
        scalers = dict()
        for k in range(len(min_max_cp)):
            scalers[k] = preprocessing.MinMaxScaler(feature_range=(0,1))
            scalers[k].fit(np.array(min_max_cp[k]).reshape(-1,1))
        
        # * Save scalars for bulk quantities 
        with open('min_max_cp.pickle','wb') as f: # min max of individual y and Cp positions
            pickle.dump(scalers,f)

        scalers = dict()
        for k in min_max.keys():
            scalers[k] = preprocessing.MinMaxScaler(feature_range=(0,1))
            scalers[k].fit(np.array(min_max[k]).reshape(-1,1))
        
        with open('min_max_scaler.pickle','wb') as f:
            pickle.dump(scalers,f) 

def create_torch_geometric_train_test_data(scraped_filename,varying_nodes=False,processed_path='processed'):
    '''
        @param scraped_filename
        @param varying_nodes = allow the number of nodes to vary
        @param use_node_labels - exports the cp values
        @processed_path - where to dump the torch.pt files
        @scale_y - scale Cl,Cd,Cdp,Cm? Default False
        @scale_cp - scale Cp by its min and max values
        @scale_y_indiv - this takes a longer time but it scales all the values of y_i by min and max of y_i
        @scale_cp_indiv - this takes a longer time but it scales all the values of cp_i by min and max of cp_i
    '''

    columns = ['xss','yss','xps','yps']
    polar_columns = ['alpha','Cl','Cd','Cdp','Cm','Reynolds','Ncrit']
    polar_columns_lists = ['Cp_ss','Cp_ps']
    df = pd.read_pickle(scraped_filename)

    # Process and Scale the data [x, y, z, conditions]
    with open('min_max_scaler.pickle','rb') as f:
        scalers = pickle.load(f)
    
    with open('min_max_y.pickle','rb') as f:
        scalers_y = pickle.load(f)

    with open('min_max_cp.pickle','rb') as f:
        scalers_cp = pickle.load(f)
    airfoils = list() # Containing all the airfoils in one list 
    for a in trange(len(df),desc='reading data into list'):
        airfoil = df.iloc[a]
        xss = airfoil['xss']        
        yss = airfoil['yss']

        xps = airfoil['xps']
        yps = airfoil['yps']
        
        x = np.concatenate((xss[0:],np.flip(xps[1:-1]))).reshape(-1,1) # This is already in 0 to 1
        y = np.concatenate((yss[0:],np.flip(yps[1:-1]))).reshape(-1,1) # 

        ones = np.ones(x.shape)
        edge_index = create_edge_adjacency(len(x))

        for p in range(len(airfoil['polars'])):            
            polar = airfoil['polars'].iloc[p]

            Cp_ss = polar['Cp_ss']
            Cp_ps = polar['Cp_ps']
            # if varying_nodes:
            #     n = np.random.randint(50,100)
            #     x_temp = np.linspace(0,1,n)
            #     yss2 = csapi(xss,yss,x_temp)                # contains the random number of points 
            #     yps2 = csapi(xps,yps,x_temp) 
                
            #     xss2 = x_temp
            #     xps2 = np.flip(xss2)

            #     x = np.concatenate((xss2[0:],np.flip(xps2[1:-1]))).reshape(-1,1)
            #     y = np.concatenate((yss2[0:],np.flip(yps2[1:-1])))
            #     y = scalers['y'].transform(y.reshape(-1,1))             # Scale all the y values

            #     ones = torch.ones(x.shape,dtype=torch.float32)
            #     edge_index = create_edge_adjacency(len(x))
            #     Cp_ss = csapi(xss,polar['Cp_ss'],x_temp)
            #     Cp_ps = csapi(xps,polar['Cp_ps'],x_temp)
          
            alpha = scalers['alpha'].transform(polar['alpha'].reshape(-1,1))[0][0]
            Reynolds = scalers['Reynolds'].transform(polar['Reynolds'].reshape(-1,1))[0][0]
            Ncrit = scalers['Ncrit'].transform(polar['Ncrit'].reshape(-1,1))[0][0]

            # Normalize Cl, Cd, Cdp, Cm
            Cl = scalers['Cl'].transform(polar['Cl'].reshape(-1,1))
            Cd = scalers['Cd'].transform(polar['Cd'].reshape(-1,1))
            Cdp = scalers['Cdp'].transform(polar['Cdp'].reshape(-1,1))
            Cm = scalers['Cm'].transform(polar['Cm'].reshape(-1,1))

            # Scale Cp
            node_labels = np.concatenate(( Cp_ss, np.flip(Cp_ps[1:-1]) ))
            node_labels = torch.as_tensor(scalers['Cp'].transform(node_labels.reshape(-1,1))[0:],dtype=torch.float32)

            data_y = torch.as_tensor(np.hstack([ Cl, Cd, Cdp, Cm ]), dtype=torch.float32)[0]
            edge_index = np.array(edge_index) # Edge Adjacency 
            if (edge_index.shape[0]!=2):
                edge_index = edge_index.transpose()
            edge_index = torch.as_tensor(edge_index,dtype=torch.long).contiguous()

            data_x = torch.as_tensor(np.hstack([y]), dtype=torch.float32)
            conditions=torch.as_tensor(np.hstack([alpha, Reynolds, Ncrit]),dtype=torch.float32)
            pos = torch.as_tensor(np.hstack([x, y]), dtype=torch.float32)
            d = Airfoil(x=data_x, conditions=conditions, edge_index=edge_index, pos=pos, y=data_y, node_labels=node_labels)
            airfoils.append(copy.deepcopy(d))

    # Load all the designs 
    random.shuffle(airfoils) # Shuffle the list
    
    train_size = int(len(airfoils)*0.7)
    test_size = len(airfoils) - train_size     

    train_subset, test_subset = random_split(airfoils,[train_size, test_size])
    train_dataset = [airfoils[i] for i in train_subset.indices] 
    test_dataset = [airfoils[i] for i in test_subset.indices] 

    train_chunks = list(chunks(train_dataset,2500))
    test_chunks = list(chunks(test_dataset,2500))
    last_indx = 0

    os.makedirs(processed_path,exist_ok=True)
    # */ Export train data in chunks /*
    train_map = list() # file_index, starting_index, last_index
    for i in trange(len(train_chunks),desc='saving the files'):
        torch.save(train_chunks[i],osp.join(processed_path,'train_{0:03d}.pt'.format(i+1)))
          
        train_map.append([i+1,last_indx,last_indx+len(train_chunks[i])-1])
        last_indx+= len(train_chunks[i])

    with open(osp.join(processed_path,'train_map.json'),'w') as f:
        json.dump(train_map,f); last_indx=0

    # */ Export test data in chunks /*
    test_map = list() # file_index, starting_index, last_index
    for i in range(len(test_chunks)):
        torch.save(test_chunks[i],osp.join(processed_path,'test_{0:03d}.pt'.format(i+1)))        
        
        test_map.append([i+1,last_indx,last_indx+len(test_chunks[i])-1])
        last_indx+= len(test_chunks[i])
    
    with open(osp.join(processed_path,'test_map.json'),'w') as f:
        json.dump(test_map,f)

def create_dataframe(scraped_filename,processed_filename, scale_y_indiv=False,scale_cp_indiv=False,drop_cp=False):
    '''
    This file optimizes the dataset from torch geometric to something more suitable for a regression
    '''
    number_of_y = -1
    number_of_cp = -1
    with open('min_max_scaler.pickle','rb') as f:
        scalers = pickle.load(f)
    
    with open('min_max_y.pickle','rb') as f:
        scalers_y = pickle.load(f)

    with open('min_max_cp.pickle','rb') as f:
        scalers_cp = pickle.load(f)
    df = pd.read_pickle(scraped_filename)
    data = list()
    for a in trange(len(df)):
        airfoil = df.iloc[a]
        xss = airfoil['xss']        
        yss = airfoil['yss']

        xps = airfoil['xps']
        yps = airfoil['yps']

        y = np.concatenate((yss[0:],np.flip(yps[1:-1]))).tolist()
        if (number_of_y<0):
            number_of_y = len(y)
        for p in range(len(airfoil['polars'])):            
            polars = airfoil['polars'].iloc[p]
            alpha = polars['alpha']
            Reynolds = polars['Reynolds']
            Ncrit = polars['Ncrit']

            Cp_ss = polars['Cp_ss']
            Cp_ps = polars['Cp_ps']
            Cp = np.concatenate(( Cp_ss, np.flip(Cp_ps[1:-1]) )).tolist()

            Cl = polars['Cl']
            Cd = polars['Cd']
            Cdp = polars['Cdp']
            Cm = polars['Cm']

            if (number_of_cp<0):
                number_of_cp = len(Cp)

            data.append(y + [alpha,Reynolds,Ncrit] + Cp + [Cl,Cd,Cdp,Cm])
            #data.append(y + [alpha,Reynolds,Ncrit] + [Cl,Cd,Cdp,Cm])


    col_y = ['y{0:02d}'.format(x) for x in range(number_of_y)] # yss to yps
    col_cond = ['alpha','Reynolds','Ncrit']
    col_Cp = ['Cp{0:02d}'.format(x) for x in range(number_of_cp)]
    col_bulk = ['Cl','Cd','Cdp','Cm']
    
    columns = col_y + col_cond + col_Cp + col_bulk 

    df = pd.DataFrame(data=data,columns=columns)

    df = df.dropna()
    # Normalize each column of y,Cl,Cd,Cdp
    if (scale_y_indiv):
        for col,n in zip(col_y,range(number_of_y)):
            df[col] = scalers_y[n].transform(df[col].to_numpy().reshape(-1,1))
    else:
        for col in col_y:
            df[col] = scalers['y'].transform(df[col].to_numpy().reshape(-1,1))

    if (scale_cp_indiv):
        for col,n in zip(col_Cp,range(number_of_cp)):
            df[col] = scalers_cp[n].transform(df[col].to_numpy().reshape(-1,1))
    else:
        for col in col_Cp:
            df[col] = scalers['Cp'].transform(df[col].to_numpy().reshape(-1,1))

    for col in col_bulk:
        df[col] = scalers[col].transform(df[col].to_numpy().reshape(-1,1))

    for col in col_cond:
        df[col] = scalers[col].transform(df[col].to_numpy().reshape(-1,1))
    if (drop_cp):
        df = df.drop(columns=col_Cp)
    df.to_pickle(processed_filename)

if __name__ == "__main__":
    # * Clean the data
    clean_scraped_data('scraped_airfoils_cp.gz')

    # * Create MinMaxScalers
    # CreateMinMaxScalers('scraped_airfoils_cp_clean.gz')

    # * Torch Geometric - Varying Nodes 
    # create_train_test_data('scraped_airfoils_cp_clean.gz',varying_nodes=True,processed_path='processed_varying_node_labels')

    # * Torch Geometric - Fixed Number of Nodes 
    create_torch_geometric_train_test_data('scraped_airfoils_cp_clean.gz',varying_nodes=False,processed_path='processed_non_varying_node_labels_v03')
   
    # create_dataframe('scraped_airfoils_cp_clean.gz','processed_non_varying_node_labels_default.gz',scale_y_indiv=True,scale_cp_indiv=True)
    
    # create_dataframe('scraped_airfoils_cp_clean.gz','processed_non_varying_node_labels_global_scale.gz',scale_y_indiv=False,scale_cp_indiv=False)

    # create_dataframe('scraped_airfoils_cp_clean.gz','processed_no_cp.gz',scale_y_indiv=False,scale_cp_indiv=False, drop_cp=True)


