
# coding: utf-8

# ## This routine processes RSP cloud data
# ## for use in Keras FC NN scheme
# #---------------------------------------------------
# 1. Read nc file (per parameter)
# 2. create Data frame from each parameter of nc
# 3. normalize each DF (Min-max) from sklearn MinMaxScaler or implicit
# 4. concat DF according to parameter inputs sets
# 5. save DF and output as .csv for further processing with Keras NN
# 6. upload .csv files for NN model processing
# 7. divide test and train
# 8. build model
# 9. compile and fit
# 10.test model (evaluate)

# In[1]:

# import neccesary modules for this module
#-------------------------------------------
import pandas as pd
import numpy as np
import scipy as sio
import scipy.stats as stats
import netCDF4 as nc4
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import shapefile
# keras related libraries
from keras.models import Sequential # import model
from keras.layers import Dense, Dropout, Activation, Flatten # import core layers
from keras.utils import np_utils # import helper funcs
from keras.regularizers import WeightRegularizer, l2 # import l2 regularizer
get_ipython().magic(u'matplotlib inline')


# In[ ]:



