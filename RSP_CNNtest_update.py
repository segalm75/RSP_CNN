
# coding: utf-8

# # Test RSP CNN with Keras

# In[1]:


# testing Keras module with RSP CNN data
# written by Michal segal Rozenhaimer, NASA Ames, Sep-2017

# upload moduls
#---------------
#general moduls
import numpy as np
import cv2
import os
import glob
import tensorflow as tf
# Keras CNN library
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape, concatenate, Concatenate
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
# import image processing module
import skimage
from skimage.io import imsave
import matplotlib.image as mpimg
from skimage import io, exposure, img_as_uint, img_as_float
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read image and check dimensions
img = io.imread('D:/ORACLES_NN/non_norm_images/ref_i/ref_i_img_000001.png')
print(img.shape)
# test


# In[3]:


# Reading ref_i, ref_q and dolp images
def read_image(image_path):
    # cv2.IMREAD_COLOR 
    # cv2.COLOR_BGR2GRAY 
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    #print("image shape", image.shape)
    #plt.imshow(image, cmap='gray')
    #plt.show()
    return np.array(image)


# In[4]:


# this function loads an image data set from one folder
# and appends it to get num_images X 1 X dim1 X dim2
# im_path is image folder path
def load_imgset(im_path) :
    data =[]
    for fname in os.listdir(im_path):
        pathname = os.path.join(im_path, fname)
        #img = Image.open(pathname)# this is with PIL library
        img = cv2.imread(pathname, cv2.IMREAD_ANYCOLOR)
        #print("img.shape",img.shape)
        img1 = img#[np.newaxis,:,:]
        #print("img1.shape",img1.shape)
        data.append(img1)
    return data


# In[5]:


ref_i_dir = 'D:/ORACLES_NN/non_norm_images/ref_i/'
ref_q_dir = 'D:/ORACLES_NN/non_norm_images/ref_q/'
dolp_dir  = 'D:/ORACLES_NN/non_norm_images/dolp/'

data_dir = 'D:/ORACLES_NN/non_norm_images/'
subdir   = ('ref_i','ref_q','dolp')
im_path     = os.path.join(data_dir,subdir[0]+'/')
print(im_path)

ref_i__= load_imgset(im_path)
ref_i_=np.array(ref_i__)
print("ref_i_ shape before squeeze",ref_i_.shape)
# cancel squeeze
ref_i = ref_i_
#ref_i = np.squeeze(ref_i_,axis=1)
#print("ref_i shape after squeeze",ref_i.shape)

# plot
fig = plt.figure()
plt.rc('ytick',labelsize=4)
plt.imshow(ref_i[194,:,:],cmap="Greys")
plt.savefig("ref_i_194_non_norm.png",bbox_inches = 'tight',dpi=1000)

print("min ref_i__non_norm_values",np.min(ref_i[194,:,:]))
print("max ref_i_non_norm_values",np.max(ref_i[194,:,:]))
#image = cv2.imread('D:/ORACLES_NN/non_norm_images/ref_i/ref_i_img_000001.png', cv2.IMREAD_ANYCOLOR)
#image.shape
# read all images in a folder
#image=read_image('D:/ORACLES_NN/non_norm_images/ref_i/ref_i_img_000001.png')
#image.shape
#training_features = np.array([read_image(im_path) for im_path in fname])
#training_features.shape
# creates a 50,32,32,3 array
#arrs = [np.random.random((32, 32, 3))
#        for i in range(50)]
# creates: 32,32,3,50 array:
# arrs_n = np.stack(arrs,axis=3)


# In[35]:


# this function loads multiple images data set from seperate folders
# and appends them to get num_images X dim1 X dim2 X num_folders/num_img_sets
# im_path is image folder path (upper level)
# im_dir is the sub-directory for a given dataset (list), e.g. ('ref_i','ref_q','dolp')
# modifications:
# MS, 2017-10-11, eliminating new axis dim and squeeze
#--------------------------------------------------------------------------------------------
def load_imgsets(im_path,im_dir) :
    # just to get num of img in folder
    path = os.path.join(im_path, im_dir[0] + '/')
    print(path)
    imset = []
    for i in range(len(os.listdir(path))):
        data = []
        # go through each of the folders in im_dir
        for d in range(len(im_dir)):
            # create img filename (e.g. ref_i_img_000001.png)
            fname = im_dir[d] + '_img_{:06d}.png'.format(i+1)
            pathname = os.path.join(im_path, im_dir[d] + '/',fname)
            print(pathname)
            img = cv2.imread(pathname, cv2.IMREAD_ANYCOLOR)
            #img1 = img[np.newaxis,:,:,]
            img1 = img#[np.newaxis,:,:,]
            data.append(img1)
        data = np.array(data)
        data = np.squeeze(data)
        # data = np.reshape(data,(7,101,len(im_dir)))
        # instead of reshape use transpose to keep meaning right:
        data = np.transpose(data,(1,2,0))
        imdata = data#[np.newaxis,:,:,:]
        imset.append(imdata)
    return imset #np.squeeze(imset)


# In[36]:


# this section merges all inputs
im_path = 'D:/ORACLES_NN/images/'
#im_path = 'D:/ORACLES_NN/non_norm_images/'
#im_dir  = ('ref_i','ref_q','dolp')
im_dir  = ('ref_i','dolp')
data_=load_imgsets(im_path,im_dir)
data = np.array(data_)
data.shape


# In[16]:


# this section defines the outputs
import pandas as pd
target_dir = 'D:/ORACLES_NN/images/targets/'
outputs = ('COD','REF','VEF')
#outputs = ('COD')
ret_params = pd.DataFrame([])
for idx, fi in enumerate(outputs):
    # read files:
    csvpath = target_dir + outputs[idx] + ".csv"
    print(csvpath)
    tmp = pd.read_csv(csvpath,index_col=False)
    ret_params = pd.concat([ret_params,tmp], axis=1)
ret_params.head()


# In[520]:


ref_i_dir = 'D:/ORACLES_NN/images/ref_i/'
ref_q_dir = 'D:/ORACLES_NN/images/ref_q/'
dolp_dir  = 'D:/ORACLES_NN/images/dolp/'

ref_i__= load_imgset(ref_i_dir)
ref_i_=np.array(ref_i__)
ref_i = np.squeeze(ref_i_,axis=1)
print("ref_i",ref_i.shape)

ref_q__= load_imgset(ref_q_dir)
ref_q_=np.array(ref_q__)
ref_q = np.squeeze(ref_q_,axis=1)
print("ref_q",ref_q.shape)

dolp__= load_imgset(dolp_dir)
dolp_=np.array(dolp__)
dolp = np.squeeze(dolp_,axis=1)
print("dolp",dolp.shape)
       


# # Test CNN for RSP

# In[39]:


# Generate data input/output
#---------------------------
np.random.seed(123)  # for reproducibility

# define input and output
ref_i_test = np.reshape(ref_i,(103680,101,7))
print(ref_i_test.shape)
X = data[:,:,:,1]#ref_i #test # 103680,7,101,3
y = ret_params['COD']  # 103680,3
print("X_shape",X.shape)
print("y_shape",y.shape)

# prepare input/output dimensions
# expand y if only 1 dim
if len(y.shape)<2:
    print("y_shape",y.shape)
    y = np.expand_dims(y,-1)
    print("y",y.shape)
# expand X if only 1 "channel"
#if len(X.shape)<4:
#    print("X_shape",X.shape)
#    X = np.expand_dims(X,-1)
#    print("X",X.shape)

# split train/test
#-----------------
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42,shuffle=True)

# print and plot data samples

print("X_shape",      X.shape)
print("X_train_shape",X_train.shape)
print("X_test_shape" ,X_test.shape)
print("X_train_shape[0]",X_train.shape[0])
print("y_train_shape",y_train.shape)
print("y_test_shape" ,y_test.shape)
print("y_train_sample",y_train[0:5])

# plot
from matplotlib import pyplot as plt
fig = plt.figure()
plt.rc('ytick',labelsize=4)
plt.imshow(X_train[194,:,:],cmap="Greys")
plt.savefig("train_194_t.png",bbox_inches = 'tight',dpi=1000)

fig = plt.figure()
plt.rc('ytick',labelsize=4)
plt.imshow(X[194,:,:],cmap="Greys")
plt.savefig("X_ref_i_194_t_255.png",bbox_inches = 'tight',dpi=1000)


# In[329]:


## prepare the model and train
##-----------------------------
# define model with Keras
#-------------------------
#You can call tf functions from Keras. from keras import backend as K and K.concat(...)

model = Sequential()
# input: 7x101 2D arrays with 3 channels -> (7, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(7, 101, 1)))
#model.add(Conv2D(1, (1,1), padding='valid', activation='relu', input_shape=(7, 101,1)))
model.add(Conv2D(input_shape=(7,101,1),kernel_size=(2,10),strides=1,
                 use_bias= True, bias_initializer='zeros',
                 activation='relu',filters=1,padding='valid', name='Conv1'))

Conv1_shape = model.layers[-1].output_shape
print("Conv1_shape",Conv1_shape)
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(input_shape=(7,101,1),kernel_size=(1,1),strides=1,
                 use_bias= True, bias_initializer='zeros',
                 activation='relu',filters=1,padding='valid', name='Conv2'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
model.add(Dense(256, activation='relu',name='fc1'))
model.add(Dropout(0.5, name='dropout_050'))
model.add(Dense(1, activation='linear',name='fc_linear'))
#model.add(Dense(10, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.001)
model.compile(loss='mean_squared_error', 
              optimizer=sgd,
              metrics=['accuracy'])

# keras.backend.shape(x) to get the shape of a tensor or:
model.summary()
#model.fit(X_train, y_train, batch_size=1000, epochs=20)
#model.fit(X_train, y_train, batch_size=1, epochs=20)

#score, acc = model.evaluate(X_test, y_test, batch_size=100)
#print('Test score:'   , score)
#print('Test accuracy:', acc  )


# In[529]:


# define model variables
batch_size = 1000
patch1_size1 = 3
patch1_size2 = 10
stride11     = 1
stride12     = 1
patch2_size1 = 3
patch2_size2 = 10
stride21     = 1
stride22     = 1
depth1       = 32 # number of filters of 1st conv layer
depth2       = 64 # number of filters of 2nd conv layer
depth3       = 60 # FC1 nodes
depth4       = 20 # FC2 nodes
shapeX       = X.shape
num_of_outputs = y.shape[1]
# expand X if only 1 "channel"
if len(X.shape)<4:
    print("X_shape",X.shape)
    X       = np.expand_dims(X,-1)
    X_train = np.expand_dims(X_train,-1)
    X_test  = np.expand_dims(X_test,-1)
    print("X",X.shape)
    print("X_train",X_train.shape)
    print("X_test",X_test.shape)

channels = X.shape[-1]
    
#if len(shapeX)<4:
#    channels = 1
#else:
#    channels = shapeX[-1]
    
image_size1   = shapeX[1]
image_size2   = shapeX[2]
print("num_of_channels",channels)
print("num_of_outputs",y.shape[1])


# In[530]:


# Size of arrays
# this matters when using padding='valid'
layer1size1 = (image_size1-patch1_size1)//stride11 + 1
layer1size2 = (image_size2-patch1_size2)//stride12 + 1
layer2size1 = (layer1size1-patch2_size1)//stride21 + 1
layer2size2 = (layer1size2-patch2_size2)//stride22 + 1
print("layer1size1",layer1size1)
print("layer1size2",layer1size2)
print("layer2size1",layer2size1)
print("layer2size2",layer2size2)


# In[531]:


# Keras functional API
#----------------------
# create the CNN layers
#----------------------
# This returns a tensor
inputs = Input(shape=(image_size1,image_size2,channels))
print("input type is:",type(inputs))
# layer Conv1
Conv1  = Conv2D(input_shape=(image_size1,image_size2,channels),kernel_size=(patch1_size1,patch1_size2),
                 strides=(stride11,stride12),
                 use_bias= True, bias_initializer='zeros',
                 kernel_initializer = TruncatedNormal(mean=0,stddev=0.1),
                 activation='relu',filters=depth1,padding='same', name='Conv1')# padding 'same' doesn't chnage layer size
conved1 = Conv1(inputs)
Conv1.output
print("Conv1_shape",Conv1.output_shape)
print("conved1_shape",conved1.shape)
# layer Conv2
Conv2  = Conv2D(kernel_size=(patch2_size1,patch2_size2),
                 strides=(stride21,stride22),
                 use_bias= True, bias_initializer='zeros',
                 kernel_initializer = TruncatedNormal(mean=0,stddev=0.1),
                 activation='relu',filters=depth2,padding='same', name='Conv2')# padding 'same' doesn't chnage layer size
conved2 = Conv2(conved1)
Conv2.output
print("Conv2_shape",Conv2.output_shape)

# 1st fully connected layer
# need to change previous output size to go into this layer
3#_in2fc1      = conved2 # input which tensor goes to fc1 layer
#print("_in2fc",_in2fc)
# reshape this tensor
#_in2fc1_shape = _in2fc1.get_shape().as_list()
#print("_in2fc_shape",_in2fc1_shape)
#in2fc1        = tf.reshape(conved1,[-1,7*101*64])# [-1] by itself flattens a tensor
#in2fc1        = tf.reshape(conved1,[-1,_in2fc1_shape[1]*_in2fc1_shape[2]*_in2fc1_shape[3]])# -1 is for batch size
#print("in2fc_shape",in2fc1.shape)

# flatten conv layer into fc input (otherwise the model can't figure out the flow dims)
flat1 = Flatten(name='flat1')
flatened = flat1(conved2)

# fully connected layer 1
fc1   = Dense(depth3, activation='relu',name='fc1')
#fced1 = fc1(in2fc1)# the layer accepts a tensor to work on
fced1 = fc1(flatened)# the layer accepts a tensor to work on
fc1.output
print("fc1_shape",fc1.output_shape)
print("fced1_shape",fced1.shape)

# fully connected layer 2
fc2   = Dense(depth4, activation='relu',name='fc2')
fced2 = fc2(fced1)
fc2.output
print("fc2_shape",fc2.output_shape)
print("fced2_shape",fced2.shape)

# add dropout to reduce overfitting
#drop1   = Dropout(rate = 0.5,name='drop1')
#droped1 = flat(fced2)

# And finally we add the main linear regression layer
lin = Dense(num_of_outputs, activation='linear',name='out_linear')
out = lin(fced2)
lin.output
print("inp_lin",lin.input_shape)
print("out_lin",lin.output_shape)
# model_output = Dense(1, activation='linear', name='model_output')(conved1)# this is a tensor not a layer...


# In[532]:


# Define model
#-------------
#model = keras.layers.add([conved1, conved2, fced1, fced2, out])
rsp_cnn_model = Model(inputs=inputs, outputs=out)
rsp_cnn_model.summary()
#rsp_cnn_model = Model(inputs=inputs, outputs=predictions)
# number of parameters per layer: (filter size1 x filter size 2 x depth (channels) + 1 (bias)) x # of filters
# example for Conv1: (3x10x1 + 1)*32 = 992
# flatten gets a 1D vector from a conv layer: e.g. 7*101*64 = 45248
# number of parameters in dense layer: flattenD (45248) * # nuerons (60 + 1 for bias) = 2714940


# In[533]:


# Complie model and train
#-------------------------
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.01)
rsp_cnn_model.compile(loss='mean_squared_error', 
              optimizer=sgd,
              metrics=['accuracy'])#loss_weights=[1., 0.2]
# early stopping condition
# stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')
# saving best model params
#checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
#hist = model.fit(..., callbacks=[checkpointer])
#model.load_weights('weights.hdf5')
#predicted = model.predict(X_test_mat)

# using callbacks to stop/save
#callbacks = [
#    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
#    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
#]

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')]

rsp_cnn_model.fit(x = X_train, y=y_train,
          epochs=2, batch_size=100,verbose=2,validation_split=0.15, callbacks = callbacks)
#score, acc = rsp_cnn_model.evaluate(X_test, y_test, batch_size=1000)


# In[352]:


# Keras functional API
# This returns a tensor
inputs = Input(shape=(7,101,1))
Conv1  = Conv2D(input_shape=(7,101,1),kernel_size=(2,10),strides=1,
                 use_bias= True, bias_initializer='zeros',
                 activation='relu',filters=1,padding='valid', name='Conv1')
conved1 = Conv1(inputs)
Conv1.output
Conv1.output_shape
# a layer instance is callable on a tensor, and returns a tensor
#x = Dense(64, activation='relu')(inputs)

#x = Dense(64, activation='relu')(x)
#predictions = Dense(10, activation='softmax')(x)

#a = Input(shape=(32, 32, 3))
#b = Input(shape=(64, 64, 3))

#conv = Conv2D(16, (3, 3), padding='same')
#conved_a = conv(a)

# first convolution layer
#conv1 = Conv2D(20, (2, 10), padding='valid', use_bias= True, bias_initializer='zeros',activation='relu')(inputs)
#conv1 = Conv2D(20, (2, 10), padding='valid', use_bias= True, bias_initializer='zeros',activation='relu')
#conv1 = Conv2D(20, (3, 3), padding='same', activation='relu')
#conv1_in = conv1(inputs)
#conv = Conv2D(16, (3, 3), padding='same')
#conved_a = conv(a)

# first pooling layer
#pol1 = MaxPooling2D(20,(1,2), padding = 'valid')(conv1)

# Only one input so far, the following will work:
#assert conv1.input_shape == (None, 7, 101, 1)
#conv1.input_shape

#x = MaxPooling2D(64, activation='relu')(x)
#predictions = Dense(10, activation='softmax')(x)

#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1, activation='linear'))



# This creates a model that includes
# the Input layer and three Dense layers
#model = Model(inputs=inputs, outputs=predictions)
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#model.fit(data, labels)  # starts training

