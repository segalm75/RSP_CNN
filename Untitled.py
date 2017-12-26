
# coding: utf-8

# In[ ]:

# Plotting ORACLES flight paths


# In[ ]:

# import neccesary modules for this module
#-------------------------------------------
import pandas as pd
import numpy as np
import scipy as sio
import scipy.stats as stats
from scipy.interpolate import griddata, interp2d
import glob
import netCDF4 as nc4
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import shapefile
import os, math, sys
import datetime as dt
import time as tm
import csv
get_ipython().magic(u'matplotlib inline')


# In[ ]:

## this function reads retrieved ".txt" files from Matlab
## given flight date, concatanates various files from same
## day if there are multiple and plots spatial maps of:
## Reff, and COD, and saves them to figure
## Michal Segal, 08-03-2017
## data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'
## subdir is 'ref_i_ref_q//' etc.
## f_date is string: yyyymmdd
## platform is aircraft: "ER-2" or "P-3"
## lat_1 - southern latitude
## lat_2 - northern latitude
## lat_0 - middle latitude
## lon_0 - middle longitude
## m_width - width of draen map in projection units (e.g., 3500000)
## m_height- height of drawn map in projection units
##---------------------------------------------------------------------------
def plotRSPlatlon_winput(data_dir,subdir,f_date,platform,lat_1,lat_2,lat_0,lon_0,m_width,m_height):
    
    # import moduls
    import numpy as np
    import scipy as sio
    from os import listdir
    from os.path import isfile, join
    from os import walk
    import pandas as pd
    import glob
    import matplotlib 
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib import cm
    from matplotlib import gridspec
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon
    from matplotlib.patches import Polygon
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import shapefile
    get_ipython().magic(u'matplotlib inline')
    
    # list all files in folder to plot
    expres = data_dir + subdir + '//' + f_date + "T*lut_scale*.txt"  
    flist = glob.glob(expres)
    print "files to process: "
    print flist
    
    ## initialize arrays
    data = pd.DataFrame([])
    # read and concatanate files:
    for f in range(len(flist)):
        print flist[f]
        if (platform =="ER-2"):
            tmp = pd.read_table(flist[f], sep='\s+', header=None,
                   names=["UTC_hr", "Alt_m", "Latitude", "Longitude", "SZA", "RelativeAzimuth","Effective_Radius",
                          "Effective_Variance","COT"])
        elif (platform =="P-3"):
             tmp = pd.read_table(flist[f], sep='\s+', header=None,
                   names=["UTC_hr", "Latitude", "Longitude", "Alt_m", "SZA", "RelativeAzimuth","Effective_Radius",
                          "Effective_Variance","COT"])
        
        print tmp.head()
        # concat arrays
        data=data.append(pd.DataFrame(tmp))
        print "tmp.shape"
        print tmp.shape
    
    print "data.shape"
    print data.shape
    
    ## plot lat/lon with COT/Reff
    ##---------------------------
    
    fig, axes = plt.subplots(1,2)
  
    projection = 'lcc'
    #llcrnrlon  = -5
    #llcrnrlat  = -30
    #urcrnrlon  = 20
    #urcrnrlat  = 5
    resolution = 'i'
    # ORACLES 2016
    #lat_1 = -30.
    #lat_2 = 5.
    #lat_0 = -15.
    #lon_0 = 5.
    m_min = lon_0 + (-10.)
    m_max = lon_0 + 15.

    parallels = np.arange(lat_1, lat_2, 5)
    meridians = np.arange(m_min, m_max, 5)
    
    # the fisrt subplot - COT
    axes[0].set_title('COT ' + f_date)
    m0 = Basemap(width = m_width, height = m_height,
                projection = projection,
                lat_1 = lat_1, lat_2 = lat_2, 
                lat_0 = lat_0, lon_0 = lon_0,
                resolution = resolution,ax=axes[0])
    m0.bluemarble(scale=0.25)
    m0.drawparallels(parallels, labels=[1,0,0,0], fontsize=8, 
       linewidth=1.)
    m0.drawmeridians(meridians, labels=[0,0,0,1], fontsize=8, 
       linewidth=1.)
   

    # the fisrt subplot - COT
    axes[1].set_title('Effective Radius ' + f_date)
    m1 = Basemap(width = m_width, height = m_height,
                projection = projection,
                lat_1 = lat_1, lat_2 = lat_2, 
                lat_0 = lat_0, lon_0 = lon_0,
                resolution = resolution,ax=axes[1])
    m1.bluemarble(scale=0.25)
    m1.drawparallels(parallels, labels=[1,0,0,0], fontsize=8, 
       linewidth=1.)
    m1.drawmeridians(meridians, labels=[0,0,0,1], fontsize=8, 
       linewidth=1.)
   
    #plt.title(f_date, fontsize=16)
    
    # read lats and lons (representing centers of grid boxes).
    lats = np.array(data['Latitude'])
    lons = np.array(data['Longitude'])
    dat0 = np.array(data['COT'])
    dat1 = np.array(data['Effective_Radius'])
    
    # create an axes on the bottom/right etc. side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider0 = make_axes_locatable(axes[0])
    #cax0 = divider0.append_axes("bottom", size="5%", pad=0.05)
    cax0 = divider0.append_axes("bottom", size="5%", pad=0.30)
    
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("bottom", size="5%", pad=0.30)

    #plt.colorbar(im, cax=cax)

    map0  = m0.scatter(lons,lats,c=dat0,cmap=plt.cm.OrRd,latlon=True)#plt.cm.gist_ncar
    cbar0 = plt.colorbar(map0, ticks=[0,5,10,15,20,25,30],cax=cax0, orientation='horizontal')
    
    map1  = m1.scatter(lons,lats,c=dat1,cmap=plt.cm.OrRd,latlon=True)
    cbar1 = plt.colorbar(map1, ticks=[0,5,10,15,20],cax=cax1, orientation='horizontal')

    #cbar.set_label(cbar_title, fontsize = 14)3 orientation='horizontal'
    plt.show()

    #fig.savefig(filepath + title + '.png')
    
    # save figure
    # as pdf
    fi1 = '../../py_figs/oracles_NN/' + 'NN_RSP_' + f_date + "_" + platform + '_cot_reff_latlon_' + subdir + '.pdf'
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(13, 9, forward=True)
    fig.savefig(fi1, bbox_inches='tight',dpi=1000)
    # as png
    fi2 = '../../py_figs/oracles_NN/' + 'NN_RSP_' + f_date + "_" + platform + '_cot_reff_latlon_' + subdir + '.png'
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(13, 9, forward=True)
    fig.savefig(fi2, bbox_inches='tight',dpi=1000)
    # or set axes interactively
    #for ax in fig.axes:
    #    ax.set_xlim(10, 20)
        
    
    
    
    return 
    

