
# coding: utf-8

# ## This routine processes RSP cloud data for ER2
# ## for use in Keras FC NN scheme
# #---------------------------------------------------------------------
# 1. Read nc file (per parameter)
# 2. create Data frame from each parameter of nc
# 3. normalize each variable (Min-max/scale) from sklearn MinMaxScaler or implicit
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
from keras.regularizers import l2 # import l2 regularizer
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## read netcdf parameters from LUT or measurements files
## nc_file is full file path, 
## e.g. filepath = '..//..//py_data//oracles_NN//training//NNcld_20150911_20160102_20160420_nonoise.nc'
## varname is netcdf variable name string in file, e.g. "ref_i", "ref_q" or "dolp"
##-------------------------------------------------------------------------------

def read_nc_file( nc_file, varname ):
    f = nc4.Dataset(nc_file)
    data = f.variables[varname]
    #f.close()
    return data;


# In[3]:


## cat netcdf parameters from LUT or measurements files
## to create dataframe in pandas, 
## colname is column name to change from default
##-----------------------------------------------------

def create_df_from_nc( nc_data, colname ):
    df = pd.DataFrame([])
    # check data size and length 
    l = len(nc_data.shape)
    vza = np.arange(24,125)
    # if l=3 cat dataframe, if l=1 save var as df
    if   l==3:
        # wavelength dim is the 2nd element
        for i in range(nc_data.shape[1]):
                #df=df.append(pd.DataFrame(nc_data[:,i,:]))
                tmp = pd.DataFrame(np.transpose(nc_data[vza,i,:]))# only VZA -40 to 40
                # rename column name
                tmp.rename(columns=lambda x: colname + '_lambda_' + str(i+1) + '_VZA_',
                               inplace=True)
                df = pd.concat([df, tmp], axis=1)
                
    elif l==2:
            tmp = pd.DataFrame(np.transpose(nc_data[3,:])) # 3 is the VZA index (all rows suppose to be similar)
            df=df.append(pd.DataFrame(tmp))
            # rename column name
            df.rename(columns=lambda x: colname, inplace=True)
    elif l==1:
            df=df.append(pd.DataFrame(nc_data[:]))
            # rename column name
            df.rename(columns=lambda x: colname, inplace=True)
    return df;


# In[9]:


## standartize data frame before applying projection
##---------------------------------------------------

def std_df( data_df ):
    
    # extract values only from DF
    data = data_df.iloc[:,:].values
    # standartize
    #if data_df.shape[1]>1:
        #data_std = StandardScaler().fit_transform(data)
    data_std = MinMaxScaler().fit_transform(data)
    #else:
        #data_std = StandardScaler().fit_transform(data.T)
        #data_std = MinMaxScaler().fit_transform(data.T)

    
    # test:
    print('original data: ', data[0:5,0:5])
    print('standartized data: ', data_std[0:5,0:5])
    
    # plot:
    #------
    # this is for DF
    #plt.plot(ref_i.iloc[1,0:151].values)
    
    if data_df.shape[1]>1:
        # orig
        plt.plot(data[10,0:100])# VZA size is 101
        plt.plot(data[10,1*100+1:2*100])
        plt.plot(data[10,2*100+1:3*100])
        plt.plot(data[10,3*100+1:4*100])
        plt.plot(data[10,4*100+1:5*100])
        plt.plot(data[10,5*100+1:6*100])
        plt.plot(data[10,6*100+1:7*100])
        plt.legend(['lambda 1 orig', 'lambda 2 orig','lambda 3 orig', 'lambda 4 orig', 
                    'lambda 5 orig', 'lambda 6 orig', 'lambda 7 orig'], loc='best')
        plt.show()

        # std
        plt.plot(data_std[10,0:100])# VZA size is 101
        plt.plot(data_std[10,1*100+1:2*100])
        plt.plot(data_std[10,2*100+1:3*100])
        plt.plot(data_std[10,3*100+1:4*100])
        plt.plot(data_std[10,4*100+1:5*100])
        plt.plot(data_std[10,5*100+1:6*100])
        plt.plot(data_std[10,6*100+1:7*100])
        plt.legend(['lambda 1 std', 'lambda 2 std','lambda 3 std', 'lambda 4 std', 
                    'lambda 5 std', 'lambda 6 std', 'lambda 7 std'], loc='best')
        plt.show()
        
    else:
        # orig
        plt.plot(np.arange(0, len(data), 1),data,'b.')
        plt.legend('orig')
        plt.show()
        # std
        plt.plot(np.arange(0, len(data), 1),data_std,'b.')
        plt.legend('std')
        plt.show()
    
    
    return pd.DataFrame(data_std, columns=data_df.columns)


# In[5]:


# 1. read rsp nc file
#--------------------
filepath = 'D://ORACLES_NN//py_data//ER2_nonoise.nc'
ref_i_ = read_nc_file( nc_file=filepath, varname="ref_i" )
ref_q_ = read_nc_file( nc_file=filepath, varname="ref_q" )
azi_   = read_nc_file( nc_file=filepath, varname="azi" )
sza_   = read_nc_file( nc_file=filepath, varname="sza" )
cod_   = read_nc_file( nc_file=filepath, varname="cod" )
ref_   = read_nc_file( nc_file=filepath, varname="sizea" )
vef_   = read_nc_file( nc_file=filepath, varname="sizeb" )
print("ref_i_shape:",ref_i_.shape)
print("ref_q_shape:",ref_q_.shape)
print("azi_shape:",azi_.shape)
print("sza_shape:",sza_.shape)
print("cod_shape:",cod_.shape)
print("ref_shape:",ref_.shape)
print("vef_shape:",vef_.shape)


# In[6]:


# 2. create DF
#==============
ref_i = create_df_from_nc( ref_i_, "ref_i" )
ref_q = create_df_from_nc( ref_q_, "ref_q" )
azi   = create_df_from_nc( azi_,   "azi" )
sza   = create_df_from_nc( sza_,   "sza" )
cod   = create_df_from_nc( cod_,   "cod" )
ref   = create_df_from_nc( ref_,   "ref" )
vef   = create_df_from_nc( vef_,   "vef" )

print(azi.shape)
print(azi.size)
print(len(azi.shape))
#print("azi:",len(azi))
#plt.plot(np.arange(0, len(azi), 1),azi.iloc[:,:].values,'b.')
#plt.show()

# replace colnames for division:
ref_q.rename(columns=lambda x: x.replace("ref_q", 'ref_i'),inplace=True)
dolp  = (ref_q / ref_i).abs()

# replace colnames back:
ref_q.rename(columns=lambda x: x.replace("ref_i", 'ref_q'),inplace=True)
dolp.rename(columns=lambda x: x.replace("ref_i", 'dolp'),inplace=True)


# concat DF to form various input combinations
#df = pd.concat([df, tmp], axis=1)
print("ref_i_df shape and data: ",ref_i.shape, ref_i.head())
print("ref_q_df shape and data: ",ref_q.shape, ref_q.head())
print("dolp_df shape and data: ", dolp.shape, dolp.head())


# In[10]:


# 3. standartize
#----------------
ref_i_std = std_df( ref_i )
ref_q_std = std_df( ref_q )
dolp_std  = std_df( dolp )
azi_std   = std_df( azi )
sza_std   = std_df( sza )
cod_std   = std_df( cod )
ref_std   = std_df( ref )
vef_std   = std_df( vef )


# In[12]:


# 4.+ 5. concat DF and save into csv
#------------------------------------
# out_norm
df_out_norm     = pd.concat([cod_std, ref_std, vef_std], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_y_train_norm.csv'
df_out_norm.to_csv(file_name, header=True, index=False)

# out_orig
df_out     = pd.concat([cod, ref, vef], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_y_train_orig.csv'
df_out.to_csv(file_name, header=True, index=False)

# data
ref_i_ref_q = pd.concat([azi_std, sza_std, ref_i_std, ref_q_std], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_x_ref_i_ref_q.csv'
ref_i_ref_q.to_csv(file_name, header=True, index=False)


ref_i_dolp = pd.concat([azi_std, sza_std, ref_i_std, dolp_std], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_x_ref_i_dolp.csv'
ref_i_dolp.to_csv(file_name, header=True, index=False)

ref_q_dolp = pd.concat([azi_std, sza_std, ref_q_std, dolp_std], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_x_ref_q_dolp.csv'
ref_q_dolp.to_csv(file_name, header=True, index=False)

ref_i_x = pd.concat([azi_std, sza_std, ref_i_std], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_x_ref_i.csv'
ref_i_x.to_csv(file_name, header=True, index=False)

ref_q_x = pd.concat([azi_std, sza_std, ref_q_std], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_x_ref_q.csv'
ref_q_x.to_csv(file_name, header=True, index=False)

dolp_x = pd.concat([azi_std, sza_std, dolp_std], axis=1)
file_name = 'D://ORACLES_NN//py_inputs//ER2_x_dolp.csv'
dolp_x.to_csv(file_name, header=True, index=False)


# In[15]:


# 6. + 7. split train/test
#---------------------------
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# run different variations
var_in = ["ref_i_ref_q","ref_i_dolp","ref_q_dolp","ref_i","ref_q","dolp"]
var_in = ["ref_i_dolp"]

for f in range(len(var_in)):
    
    # import x
    x_filename = 'D://ORACLES_NN//py_inputs//ER2_x_' + var_in[f] + '.csv'
    X = pd.read_csv(x_filename)
    #print(X.head())
    #import y
    y = pd.read_csv('D://ORACLES_NN//py_inputs//ER2_y_train_orig.csv')
    #print(y.head())
    # import train/validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42,shuffle=True)

    # print and plot data samples

    print("X_shape",      X.shape)
    print("X_train_shape",X_train.shape)
    print("X_test_shape" ,X_test.shape)
    print("X_train_shape[0]",X_train.shape[0])
    print("y_train_shape",y_train.shape)
    print("y_test_shape" ,y_test.shape)
    print("y_train_sample",y_train[0:5])


# In[17]:


# 8. build model
#-----------------

