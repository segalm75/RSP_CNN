
# coding: utf-8

# # Plots of retrieved RSP cloud parametrs using NN scheme

# In[1]:

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


# In[18]:

## this function reads retrieved ".txt" files from Matlab
## given flight date, concatanates various files from same
## day if there are multiple and plots timeseries of:
## Reff, Veff, COD, and saves them to figure
## Michal Segal, 05-07-2017
## data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'
## subdir is subdirectory of retrieved files (by ref_i_ref_q etc.)
## f_date is string: yyyymmdd
## platform is aircraft: "ER-2" or "P-3"
##---------------------------------------------------------------------------
def plotRSPtimeseries(data_dir,subdir,f_date,platform):
    
    # import moduls
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
            
        print tmp.shape
        print tmp.head()
        # concat arrays
        data=data.append(pd.DataFrame(tmp))
        print "tmp.shape"
        print tmp.shape
    
    print "data.shape"
    print data.shape
    
    # save concat data into .csv file by date
    file_name = data_dir + subdir + '//' + 'NN_RSP_' + f_date + platform + '_' + subdir + ".csv"
    print 'file2save is: ' + file_name
    
    data.to_csv(file_name, header=True, index=False)
    
    ## plot timeseries using all concat files
    plt.figure()
    #ax1 = plt.subplot(3, 1, 1)
    #plt.plot(data['UTC_hr'],data['Effective_Radius'], 'o')
    #ax2 = plt.subplot(3,1,2)
    #plt.plot(data['UTC_hr'],data['Effective_Variance'], 'o')
    #ax3 = plt.subplot(3,1,3)
    #plt.plot(data['UTC_hr'],data['COT'], 'o')
    
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0, 1, 3)]
    
    # set height ratios for sublots
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1]) 

    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    line0, = ax0.plot(data['UTC_hr'],data['Effective_Radius'], '.', color='r')
    ax0.set_title(f_date, fontsize = 14)
    #pyplot.locator_params(axis = 'x', nbins = 4)
    # plt.locator_params(axis = 'y',numticks=4)
    ax0.set_yticks([5,10,15,20])
    
    #the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    # log scale for axis Y of the first subplot
    ax1.set_yscale("log")
    line1, = ax1.plot(data['UTC_hr'],data['Effective_Variance'], '.', color='b')
    ax1.set_yticks([1e-3,1e-2,1e-1])
    
    #the third subplot
    # shared axis X
    ax2 = plt.subplot(gs[2], sharex = ax0)
    line2, = ax2.plot(data['UTC_hr'],data['COT'], '.', color='g')
    ax2.set_xlabel('UTC [hr]', fontsize = 12)
    ax2.set_yticks([0,10,20,30,40])
    
    #ax.xaxis.set_label_coords(1.05, -0.025)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    # put lened on first subplot
    ax1.legend((line0, line1, line2), ('Effective Radius', 'Effective Variance', 'COT'), loc='lower left')

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    #plt.show()

    
    # save figure
    # as pdf
    fi1 = '../../py_figs/oracles_NN/' + 'NN_RSP_' + f_date + "_" + platform + '_reff_veff_cot_timeseries_' + subdir + '.pdf'
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(13, 9, forward=True)
    plt.savefig(fi1, bbox_inches='tight',dpi=1000)
    # as png
    fi2 = '../../py_figs/oracles_NN/' + 'NN_RSP_' + f_date + "_" + platform + '_reff_veff_cot_timeseries_' + subdir + '.png'
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(13, 9, forward=True)
    plt.savefig(fi2, bbox_inches='tight',dpi=1000)
    # or set axes interactively
    #for ax in fig.axes:
    #    ax.set_xlim(10, 20)
        
    return flist
    


# In[4]:

# test plotRSPtimeseries(data_dir,f_date):
#
#data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'
#plotRSPtimeseries(data_dir,"20160927","ER-2")

#data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut0//'
#plotRSPtimeseries(data_dir,"ref_q_dolp","20170801","P-3")
#flist[0]
# [utc, alt, lat,lon,sza,relAzi,Reff,Veff,COD];
#tmp = pd.read_table('..\\..\\py_data\\oracles_NN\\input2NN\\predictions\\lut9\\20160910T095856Z_V001-20160912T173255Z_NN_ref_i_dolp_090_ref_i_dolp_090_prediction_lut_scale_with_lut_9.txt',
                    #sep='\s+', header=None,
                    #names=["UTC_hr", "Alt_m", "Latitude", "Longitude", "SZA", "RelativeAzimuth","Reff","Veff","COT"])
#tmp.head()


# In[19]:

# loop over all campaign dates to get plots
flight_dates = ['20160910','20160912', '20160914', '20160916','20160918','20160920','20160922', '20160924','20160925', '20160927']
data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'

for f in range(len(flight_dates)):
    plotRSPtimeseries(data_dir,"ref_q_dolp",flight_dates[f],"ER-2")


# In[7]:

# loop over all campaign dates to get plots
#flight_dates = ['20170815','20170817','20170818','20170819']
#flight_dates = ['20170801','20170807','20170809','20170812','20170813']
#flight_dates = ['20170824']
#flight_dates = ['20170821']
#flight_dates = ['20170826','20170828']
flight_dates = ['20170830','20170831','20170902']
data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut0//'

for f in range(len(flight_dates)):
    plotRSPtimeseries(data_dir,"ref_i_ref_q",flight_dates[f],"P-3")


# In[8]:

## this function reads retrieved ".txt" files from Matlab
## given flight date, concatanates various files from same
## day if there are multiple and plots spatial maps of:
## Reff, and COD, and saves them to figure
## Michal Segal, 05-07-2017
## data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'
## f_date is string: yyyymmdd
## platform is aircraft: "ER-2" or "P-3"
##---------------------------------------------------------------------------
def plotRSPlatlon(data_dir,f_date,platform):
    
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
    expres = data_dir + f_date + "T*lut_scale*.txt"  
    flist = glob.glob(expres)
    print "files to process: "
    print flist
    
    ## initialize arrays
    data = pd.DataFrame([])
    # read and concatanate files:
    for f in range(len(flist)):
        print flist[f]
        tmp = pd.read_table(flist[f], sep='\s+', header=None,
               names=["UTC_hr", "Alt_m", "Latitude", "Longitude", "SZA", "RelativeAzimuth","Effective_Radius",
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
    llcrnrlon  = -5
    llcrnrlat  = -30
    urcrnrlon  = 20
    urcrnrlat  = 5
    resolution = 'i'
    lat_1 = -30.
    lat_2 = 5.
    lat_0 = -15.
    lon_0 = 5.

    parallels = np.arange(-30, 5, 5)
    meridians = np.arange(-5, 20, 5)
    
    # the fisrt subplot - COT
    axes[0].set_title('COT ' + f_date)
    m0 = Basemap(width = 2500000, height = 3500000,
                projection = projection,
                lat_1 = lat_1, lat_2 = lat_2, 
                lat_0 = lat_0, lon_0 = lon_0,
                resolution = resolution,ax=axes[0])
    m0.bluemarble(scale=0.25)
    m0.drawparallels(parallels, labels=[1,0,0,0], fontsize=12, 
       linewidth=1.)
    m0.drawmeridians(meridians, labels=[0,0,0,1], fontsize=12, 
       linewidth=1.)
   

    # the fisrt subplot - COT
    axes[1].set_title('Effective Radius ' + f_date)
    m1 = Basemap(width = 2500000, height = 3500000,
                projection = projection,
                lat_1 = lat_1, lat_2 = lat_2, 
                lat_0 = lat_0, lon_0 = lon_0,
                resolution = resolution,ax=axes[1])
    m1.bluemarble(scale=0.25)
    m1.drawparallels(parallels, labels=[1,0,0,0], fontsize=12, 
       linewidth=1.)
    m1.drawmeridians(meridians, labels=[0,0,0,1], fontsize=12, 
       linewidth=1.)
   
    #plt.title(f_date, fontsize=16)
    
    # read lats and lons (representing centers of grid boxes).
    lats = np.array(data['Latitude'])
    lons = np.array(data['Longitude'])
    dat0 = np.array(data['COT'])
    dat1 = np.array(data['Effective_Radius'])
    
    # create an axes on the right/bottom side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("bottom", size="5%", pad=0.05)
    
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("bottom", size="5%", pad=0.05)

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
    fi1 = '../../py_figs/oracles_NN/' + f_date + "_" + platform + '_cot_reff_latlon.pdf'
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(13, 9, forward=True)
    fig.savefig(fi1, bbox_inches='tight',dpi=1000)
    # as png
    fi2 = '../../py_figs/oracles_NN/' + f_date + "_" + platform + '_cot_reff_latlon.png'
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(13, 9, forward=True)
    fig.savefig(fi2, bbox_inches='tight',dpi=1000)
    # or set axes interactively
    #for ax in fig.axes:
    #    ax.set_xlim(10, 20)
        
    
    
    
    return 
    


# In[10]:

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
    


# In[13]:

#ORACLES 2016
#data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'
#plotRSPlatlon(data_dir,"20160927","ER-2")

#data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'
#plotRSPlatlon_winput(data_dir,"20160927","ER-2",-30.,5.,15.,5.,2500000,3500000)

#ORACLES 2017
data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut0//'
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170801","P-3",10.,30.,15.,-70.,4000000,3000000)
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170807","P-3",-20.,15,0,-40.,5000000,3500000)# transit
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170809","P-3",-20.,5.,-10,-5.,3000000,3500000)# including ASI
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170812","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170813","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170815","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170817","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170818","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170819","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170821","P-3",-20.,5.,-10,-5.,3000000,3500000)# including ASI
#plotRSPlatlon_winput(data_dir,"ref_i_ref_q","20170824","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170826","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
#plotRSPlatlon_winput(data_dir,"ref_q_dolp","20170828","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
plotRSPlatlon_winput(data_dir,"ref_i_ref_q","20170830","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
plotRSPlatlon_winput(data_dir,"ref_i_ref_q","20170831","P-3",-20.,5.,-10,5.,2500000,3500000)# Routine/TMS
plotRSPlatlon_winput(data_dir,"ref_i_ref_q","20170902","P-3",-20.,5.,-10,-5.,3000000,3500000)# including ASI


# In[65]:

# loop over all campaign dates to get plots
flight_dates = ['20160910','20160912', '20160914', '20160922', '20160924', '20160927']
data_dir = '..//..//py_data//oracles_NN//input2NN//predictions//lut9//'

for f in range(len(flight_dates)):
    plotRSPlatlon(data_dir,flight_dates[f],"ER-2")


# In[ ]:



