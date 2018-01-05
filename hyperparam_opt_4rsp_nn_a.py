from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.utils import eval_hyperopt_space
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils
import numpy, json
import globalvars
import pandas
import matplotlib.pyplot as plt
from   datetime import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
 
 
def data():
    '''
    Data providing function:
 
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    pre_file = 'D://ORACLES_NN//py_inputs//ER2_x_'
    var_in   = "ref_i_dolp"
    var_out  = 'D://ORACLES_NN//py_inputs//ER2_y_train_orig.csv'
    x_filename = pre_file + var_in + '.csv'
    X_ = pandas.read_csv(x_filename)
    
    Y_ = pandas.read_csv(var_out)

    X = X_.iloc[:,:].values
    Y = Y_.iloc[:,:].values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42,shuffle=True)

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    Y_train = Y_train.astype('float32')
    Y_test  = Y_test.astype('float32')
    
    
    return X_train, X_test, Y_train, Y_test

 
def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:
 
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    
    model = Sequential()

    model.add(Dropout({{uniform(0, 1)}}, input_shape=X_train.shape[1:], name="drop_input"))
    model.add(Dense({{choice([256, 512, 1024])}},  name="fc1"))
    model.add(Activation({{choice(['relu', 'tanh'])}}))
    model.add(Dropout({{uniform(0, 1)}}, name="drop_fc1"))
    model.add(Dense({{choice([256, 512, 1024])}}, name="fc2"))
    model.add(Activation({{choice(['relu', 'tanh'])}}))
    model.add(Dropout({{uniform(0, 1)}}, name="drop_fc2"))


    if conditional({{choice(['two', 'three'])}}) == 'three':
        model.add(Dense({{choice([256, 512, 1024])}}, name="fc3"))
        model.add(Activation({{choice(['relu', 'tanh'])}}))
        model.add(Dropout({{uniform(0, 1)}}, name="drop_fc3"))


    
    model.add(Dense(Y_test.shape[1], activation='linear',name='out_linear'))

    model.summary()
    

         
    choiceval = {{choice(['adam','sgd'])}}
    if choiceval == 'adam':
        adam    = Adam(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, 
                                decay={{choice([1e-2,1e-3,1e-4])}})
        optim = adam
    elif choiceval == 'rmsprop':
        rmsprop = RMSprop(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, 
                                decay={{choice([1e-2,1e-3,1e-4])}})
        optim = rmsprop
    else:
        sgd     = SGD(lr={{choice([10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])}}, 
                                decay={{choice([1e-2,1e-3,1e-4])}})

        optim = sgd


    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])

    globalvars.globalVar += 1

    var_in   = "ref_i_dolp"

    filepath = "D:/ORACLES_NN/py_outputs/weights_rsp_nn_hyperas" + var_in + str(globalvars.globalVar) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    csv_logger = CSVLogger('D:/ORACLES_NN/py_outputs/hyperas_rsp_nn_log_' + var_in + '.csv', 
                                   append=True, separator=';')
    tensor_board_logfile = 'D:/ORACLES_NN/py_logs/' + var_in + str(globalvars.globalVar)
    tensor_board = TensorBoard(log_dir=tensor_board_logfile, histogram_freq=0, write_graph=True)


    if 'results' not in globals():
        global results
        results = []

    history = model.fit(X_train, Y_train,
              batch_size={{choice([64, 128, 256, 512])}},
              nb_epoch=30,
              verbose=2,
              validation_data=(X_test, Y_test),
              callbacks=[checkpoint,csv_logger,tensor_board])

    print(history.history.keys())
    

    h1   = history.history
    acc_ = numpy.asarray(h1['acc'])
    loss_ = numpy.asarray(h1['loss'])
    val_loss_ = numpy.asarray(h1['val_loss'])
    val_acc_  = numpy.asarray(h1['val_acc'])
    parameters = space
    opt        = numpy.asarray(parameters["choiceval"])
    if choiceval == 'adam':
      lr         = numpy.asarray(parameters["lr"])
      decay      = numpy.asarray(parameters["decay"])
    elif choiceval == 'rmsprop':
      lr         = numpy.asarray(parameters["lr_1"])
      decay      = numpy.asarray(parameters["decay_1"])
    elif choiceval == 'sgd':
      lr         = numpy.asarray(parameters["lr_2"])
      decay      = numpy.asarray(parameters["decay_2"])

    results.append(parameters)


    acc_plot = 'D:/ORACLES_NN/py_plots/accuracy_run_' + var_in + str(globalvars.globalVar) + ".png"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('run: ' + var_in + str(globalvars.globalVar) + " opt: " + str(opt) + " lr: " + str(lr) + " decay: " + str(decay))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_plot)   
    plt.close()  
    
    los_plot = 'D:/ORACLES_NN/py_plots/losses_run_' + var_in + str(globalvars.globalVar) + ".png"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('run: ' + var_in + str(globalvars.globalVar) + " opt: " + str(opt) + " lr: " + str(lr) + " decay: " + str(decay))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(los_plot)   
    plt.close()  
    

    
    acc_and_loss = numpy.column_stack((acc_, loss_, val_acc_, val_loss_))
    save_file_mlp = 'D:/ORACLES_NN/py_outputs/rsp_nn_run_' + var_in + '_' + str(globalvars.globalVar) + '.txt'
    with open(save_file_mlp, 'w') as f:
            numpy.savetxt(save_file_mlp, acc_and_loss, delimiter=",")


    score, acc = model.evaluate(X_test, Y_test, verbose=0)

    print("parameters for run " + var_in + str(globalvars.globalVar) + ":")
    print("-------------------------------")
    print(parameters)
    print("opt: ", opt)
    print("lr: ", lr)
    print("decay: ", decay)
    print('Test accuracy:', acc)

    save_file_params = 'D:/ORACLES_NN/py_outputs/params_run_' + var_in + '_' + str(globalvars.globalVar) + '.txt'
    rownames  = numpy.array(['Input', 'Run', 'optimizer', 'learning_rate', 'decay', 'train_accuracy','train_loss','val_accuracy', 'val_loss', 'test_accuracy'])
    rowvals   = (var_in, str(globalvars.globalVar), opt, lr, decay, acc_[-1], loss_[-1], val_acc_[-1], val_loss_[-1],acc)

    DAT =  numpy.column_stack((rownames, rowvals))
    numpy.savetxt(save_file_params, DAT, delimiter=",",fmt="%s")

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
 
if __name__ == '__main__':

    var_in   = "ref_i_dolp"
    trials=Trials()
    best_run, best_model, space = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=300,
                                          trials=trials,
                                          eval_space=True,
                                          return_space=True)
    X_train, X_test, Y_train, Y_test = data()
    print("Evalutation of best performing model:")
    print("Parameters of best run", best_run)
    #real_param_values = eval_hyperopt_space(space, best_run)
    #print("Parameters of best run", real_param_values)
    print(best_model.evaluate(X_test, Y_test))
    json.dump(best_run, open("D:/ORACLES_NN/py_outputs/best_nn_rsp_run_" + var_in + ".txt", 'w'))

    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        print("Trial %s vals: %s" % (t, vals))
        print(eval_hyperopt_space(space, vals))
