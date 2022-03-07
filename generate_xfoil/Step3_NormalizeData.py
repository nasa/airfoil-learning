'''
    This script normalizes the data in two ways, using min-max and mean-(standard deviation)
    
'''

import pickle
from sklearn import preprocessing
from tqdm import trange
import numpy as np
import json
import os.path as osp 
import glob 

def CreateScalers(data_folder:str='json'):
    """Reads all the data and determines the min and max value. 
        This is used to create the minmaxscalers for normalizing the dataset. 
        The outcome of this function is a pickle file containing the weights for sklearn minmax scaler

    Args:
        data_folder (str): Location of all the json files 

    """

    data_files = glob.glob(osp.join(data_folder,'*.json'))
    jsons = list() 
    for filename in data_files:
        with open(filename,'r') as f:
            jsons.append(json.load(f))

    polar_columns = ['alpha','Cl','Cd','Cdp','Cm','Re','Ncrit']
    polar_columns_lists = ['Cp_ss','Cp_ps']


    # Lets find a happy medium to normalize the data by selecting random data 
    min_max = dict() # min_max of all     
    standard_scaler_data = dict() # This contains the mean and standard deviation of the dataset 
    
    cp_stacked = list() # this holds a matrix of all the [[Cp],[Cp]] allows independent min/max of Cp at each x coordinate 

    for a in trange(len(jsons),desc='reading json data'): # loop for all the airfoils
        airfoil = jsons[a]
        '''
            Find min-max for yss and yps. xss and xps should be between 0 and 1 and won't need to be scaled 
        '''
        for col in ['yss','yps']:                                         # min_max for xss,yss,xps,yps            
            if col not in min_max.keys():
                min_max[col] = [min(airfoil[col]), max(airfoil[col])]
                standard_scaler_data[col] = list()
                standard_scaler_data[col] = airfoil[col]  # This can be a lot of data 
            else:
                if min(airfoil[col]) < min_max[col][0]:
                    min_max[col][0] = min(airfoil[col])
                if max(airfoil[col]) > min_max[col][1]:
                    min_max[col][1] = max(airfoil[col])
                standard_scaler_data[col].extend(airfoil[col])  # This can be a lot of data 
        '''
            Find min-max of polars ['alpha','Cl','Cd','Cdp','Cm','Re','Ncrit']
        '''
        for polar in airfoil['polars']:
            for col in polar_columns:   # Since these are scalers we can simply append them to make a big list 
                if col not in min_max.keys():
                    min_max[col] = list()
                    min_max[col].append(polar[col])
                    standard_scaler_data[col] = list()
                    standard_scaler_data[col].append(polar[col])  
                else:
                    min_max[col].append(polar[col])
                    standard_scaler_data[col].append(polar[col])

                '''
                    For each polar there's a list of Cp for suction side and pressure side. Find the min-max of that.
                '''
            
            for col in polar_columns_lists:
                if col not in min_max.keys():
                    min_max[col] = [min(polar[col]), max(polar[col])]                    
                    standard_scaler_data[col] = list()
                    standard_scaler_data[col].extend(polar[col])  
                else:
                    if min(polar[col]) < min_max[col][0]:
                        min_max[col][0] = min(polar[col])
                    if max(polar[col]) > min_max[col][1]:
                        min_max[col][1] = max(polar[col])
                    standard_scaler_data[col].extend(polar[col])
            
            cp_stacked.append(polar['Cp_ss'])
            polar['Cp_ps'].reverse()
            cp_stacked.append(polar['Cp_ps']) # This stacked array keeps track of the Cp value at a single x coordinate, this way we can later find the min and max at x = 0.2c etc. Note: x is normalized with the chord so x is really x/c and goes from 0 to 1



    min_max['y'] = [ min([min_max['yss'][0], min_max['yps'][0]]), max([min_max['yss'][1], min_max['yps'][1]]) ]
    min_max['Cp'] = [ min([min_max['Cp_ss'][0], min_max['Cp_ps'][0]]), max([min_max['Cp_ss'][1], min_max['Cp_ps'][1]]) ]
    min_max.pop('Cp_ss', None)  # remove key from dictionary, use Cp and not individual Cp_ss and Cp_ps 
    min_max.pop('Cp_ps', None)
    min_max.pop('yss', None)    # remove key from dictionary, use y and not individual yss and yps 
    min_max.pop('yps', None)

    standard_scaler_data['y'] = standard_scaler_data['yss'] + standard_scaler_data['yps'] 
    standard_scaler_data['Cp'] = standard_scaler_data['Cp_ss'] + standard_scaler_data['Cp_ps'] 

    standard_scaler_data.pop('Cp_ss', None) # remove key from dictionary, use Cp and not individual Cp_ss and Cp_ps 
    standard_scaler_data.pop('Cp_ps', None)
    standard_scaler_data.pop('yss', None)   # remove key from dictionary, use y and not individual yss and yps 
    standard_scaler_data.pop('yps', None)
    # Note: there maybe some cases where Cp on suction side value is extremely low. We need to check for those. Anything below -2.5 is questionable. 

    
    '''
        Use sklearn to create minmax scalers for all possible columns in ['yss','yps','alpha','Cl','Cd','Cdp','Cm','Reynolds','Ncrit','Cp_ss','Cp_ps']
    '''
    min_max_scalers = dict()        
    keys = list(min_max.keys())
    for k in keys:
        min_max_scalers[k] = preprocessing.MinMaxScaler(feature_range=(0,1))
        min_max_scalers[k].fit(np.array(min_max[k]).reshape(-1,1))

    standard_scalers = dict()
    keys = list(standard_scaler_data.keys())
    for k in keys:
        standard_scalers[k] = preprocessing.StandardScaler()                
        standard_scalers[k].fit(np.array(standard_scaler_data[k]).reshape(-1,1))

    # min_max['Cp'] = [-2.5,1.2] # most data is around this
    cp_stacked = np.array([np.array(xi) for xi in cp_stacked])
    cp_min_max_scalers = list()
    for i in range(cp_stacked.shape[1]): # Look at all the Cp in each x value
        cp_min_max_scalers.append(preprocessing.MinMaxScaler(feature_range=(0,1)))
        cp_min_max_scalers[-1].fit(cp_stacked[i,:].reshape(-1,1))

    cp_standard_scalers = list()
    for i in range(cp_stacked.shape[1]): # Look at all the Cp in each x value
        cp_standard_scalers.append(preprocessing.MinMaxScaler(feature_range=(0,1)))
        cp_standard_scalers[-1].fit(cp_stacked[i,:].reshape(-1,1))

    """
        Standard vs standard cp: When you scale Cp you can scale at each x coordinate think of putting vertical bars at x = 0.2 for every graph and that's your min_max/standard scaler. The other method is to look at the value of Cp for all x for all airfoils and scale. This is what min_max and standard scalers do. This wont give as good result as min_max_cp and standard_cp 
    """
    with open('scalers.pickle','wb') as f: # min max of individual y and Cp positions
        pickle.dump({'min_max':min_max_scalers, 
                'standard':standard_scalers,
                'min_max_cp':cp_min_max_scalers,
                'standard_cp':cp_standard_scalers},f)

if __name__ == "__main__":
    CreateScalers(data_folder='json_cp_resize')