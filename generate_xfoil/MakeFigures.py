import torch
import matplotlib.pyplot as plt
import glob
import os.path as osp
import numpy as np 
import pandas as pd
from utils import csapi

# Folders 
data = pd.read_pickle('scraped_airfoils_cp_clean.gz')
airfoil_names = data['name'].unique()
name = airfoil_names[np.random.randint(0,len(airfoil_names))]
print(name)

airfoil = data[data.name==name]
xss = np.array(airfoil['xss'])[0]; yss = np.array(airfoil['yss'])[0]; xps = np.array(airfoil['xps'])[0]; yps = np.array(airfoil['yps'])[0]        

xv = np.linspace(0,1,20) # ! This fixes the number of points

yss_v = csapi(xss,yss,xv)
yps_v = csapi(xps,yps,xv)

x = np.concatenate( (xss, np.flip(xps[1:-1])) )
y = np.concatenate( (yss, np.flip(yps[1:-1])) )

polars = airfoil['polars'].iloc[0]
polars = polars[polars.Reynolds==100000]
polars = polars[polars.Ncrit==5]
alpha = polars.alpha
Cl = polars.Cl
Cp = np.concatenate(( polars.iloc[50].Cp_ss, np.flip( polars.iloc[50].Cp_ps[1:-1]) ))

    
# # Plot the Airfoil
# fig,ax = plt.subplots(1,1)
# x = np.concatenate((xss,np.flip(xps[1:-1])))
# y = np.concatenate((yss,np.flip(yps[1:-1])))

# x2 = np.concatenate((xv,np.flip(xv[1:-1])))
# y2 = np.concatenate((yss_v,np.flip(yps_v[1:-1])))
# # ax.plot(x,y, color='black', linestyle='solid', linewidth=2,marker='o', label='{0} points'.format(len(x)))
# ax.plot(x2,y2, color='red', linestyle='dashed', linewidth=4,marker='^', markersize=12,label='{0} points'.format(len(x2)))
# ax.set(xlim=(-0.05,1.05), ylim=(-0.15, 0.15))
# ax.legend(loc='lower right')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_aspect('equal')
# fig.canvas.draw()
# fig.canvas.flush_events()
# plt.show()

# fig,ax = plt.subplots(1,1)
# # Plot Cl vs Alpha for a given Reynolds, Ncrit
# ax.plot(alpha,Cl, color='black', linestyle='solid', linewidth=2, label=name)
# ax.legend(loc='lower right')
# ax.set_xlabel('alpha [deg]')
# ax.set_ylabel('Cl')

# fig.canvas.draw()
# fig.canvas.flush_events()
# plt.show()

fig,ax = plt.subplots(1,1)
# Plot Cl vs Alpha for a given Reynolds, Ncrit
ax.plot(x,Cp, color='blue', linestyle='None', linewidth=2,marker='^', markersize=5, label=name)
ax.invert_yaxis()
ax.legend(loc='lower right')
ax.set_xlabel('x/c')
ax.set_ylabel('Cp')

fig.canvas.draw()
fig.canvas.flush_events()
plt.show()