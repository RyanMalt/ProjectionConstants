from options import getVecData, getConstData
import keras
from keras import losses, regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import matplotlib
matplotlib.use('Agg') #If you can't plot the history, comment this line out
import matplotlib.pyplot as plt

import random as rn
import tensorflow as tf
import ast

import os
import datetime
import smtplib
from email.mime.text import MIMEText

import numpy as np

def generate_model(args):
    '''Accepts dictionary of attributes, then creates and trains model'''
    
    #Set the RNG seed for consistent results 
    if args['reproducible']:
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)
        rn.seed(12345)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(1234)

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    #Dimension of R and subspace
    n = args['input_size'][0]
    m = args['input_size'][1]
    input_dim = n*m

    #Load data
    trainVecData = getVecData(n, m, args['train_points'], args['train_points_version'])
    trainConstData = getConstData(n, m, args['train_points'], args['train_points_version'])

    testVecData = getVecData(n, m, args['test_points'], args['test_points_version']) 
    testConstData = getConstData(n, m, args['test_points'], args['test_points_version'])

    #Allows for stacking of augments
    #Within each augmentation, the index variable prevents overwriting of columns
    total_augs = 0

    #Preprocess data and augment inputs
    if args['augment_lewicki']:
        for i in range(n*m):
            index = total_augs + (total_augs + 2)*i
            trainVecData = np.insert(trainVecData, index, trainVecData[:, index]/(1 - 2*trainVecData[:, index]), axis=1)
            testVecData = np.insert(testVecData, index, testVecData[:, index]/(1-2*testVecData[:, index]), axis=1)

        #Allows for multiple augments
        total_augs += 1
        input_dim = n*m*(total_augs + 1)
    
    if args['augment_division']:
        for i in range(n*m):
            index = total_augs + (total_augs + 2)*i
            trainVecData = np.insert(trainVecData, index, 1/trainVecData[:, index], axis=1)
            testVecData = np.insert(testVecData, index, 1/testVecData[:, index], axis=1)

        #Allows for multiple augments
        input_dim = n*m*(total_augs + 1)
        total_augs += 1
    
    if args['augment_zeros']:
        zeros_train = np.zeros(trainVecData.shape[0])
        zeros_test = np.zeros(testVecData.shape[0])
        for i in range(n*m):
            index = total_augs + (total_augs + 2)*i
            trainVecData = np.insert(trainVecData, index, zeros_train, axis=1)
            testVecData = np.insert(testVecData, index, zeros_test, axis=1)

        #Allows for multiple augments
        total_augs += 1
        input_dim = n*m*(total_augs + 1)



    if args['random']:
        np.random.shuffle(trainVecData)

    #Initialize and setup model
    model = Sequential()

    #Add layers to the model
    model.add(Dense(args['network_layers'][0], activation=args['activation_function'], bias_regularizer=regularizers.l1(args['bias_regularization']), activity_regularizer=regularizers.l1(args['activation_regularization']), kernel_initializer='he_normal', bias_initializer='he_normal', input_dim=input_dim))
    for size in args['network_layers'][1:-1]:
        model.add(Dense(size, activation=args['activation_function'], bias_regularizer=regularizers.l1(args['bias_regularization']), activity_regularizer=regularizers.l1(args['activation_regularization']), kernel_initializer='he_normal', bias_initializer='he_normal'))
    model.add(Dense(args['network_layers'][-1], activation=None, bias_regularizer=regularizers.l1(args['bias_regularization']), activity_regularizer=regularizers.l1(args['activation_regularization']), kernel_initializer='he_normal', bias_initializer='he_normal'))

    #Initialize optimizer variable
    optimizer = None

    #Select optimizer based on command line args
    if args['optimizer'] == 'sgd':
        optimizer = SGD(lr=args['learning_rate'], decay=1e-6, momentum=.9, nesterov=True)
    elif args['optimizer'] == 'rmsprop':
        optimizer = RMSprop(lr=args['learning_rate'], rho=.9, epsilon=None, decay=0)
    elif args['optimizer'] == 'adam':
        optimizer = Adam(lr=args['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)
    else:
        print('ERROR: Invalid optimizer.  Commandline option parsing is bugged.')
        exit(1)


    #Compile model
    model.compile(loss=args['loss_function'],
                  optimizer=optimizer,
                  metrics=[args['metric']])

    #Callback system
    callbacks = None

    #Add callbacks to list
    if args['early_stopping']:
        callbacks = [EarlyStopping(patience=int(args['early_stopping'][0]), min_delta=args['early_stopping'][1], mode='min')]
    
    #Load up tensorboard information
    if args['tensorboard_file']:
        tb_args = None

        with open(args['tensorboard_file'], 'r') as f:
            s = f.read()
            tb_args = ast.literal_eval(s)
            tb =  TensorBoard(tb_args['log_dir'], tb_args['histogram_freq'], tb_args['batch_size'],
                            tb_args['write_graph'], tb_args['write_grads'], tb_args['write_images'],
                            tb_args['embeddings_freq'], tb_args['embeddings_layer_names'],
                            tb_args['embeddings_metadata'], tb_args['embeddings_data'],
                            tb_args['update_freq'])

        if callbacks:
           callbacks.append(tb)
        else:
           callbacks = [tb]
    
    if args['save_weights']:
        filepath = os.path.join('..', 'Weights', args['save_weights'])
        ms = ModelCheckpoint(filepath=filepath, verbose=args['verbose'], save_best_only=True, save_weights_only=True)

        if callbacks:
            callbacks.append(ms)
        else:
            callbacks = [ms]

    #Example code for adding additional callbacks to list
    #if args['callback']:
    #   if callbacks:
    #       callbacks.append(MakeNewCallback(args['callback'])
    #   else:
    #       callbacks = [MakeNewCallback(args['callback'])
    #
    #NOTE: You should be able to just copy and paste this block of code for any additional callbacks, just also be sure to add it to the options.py file
    
    #Train model
    model_history = model.fit(x = trainVecData, y = trainConstData, validation_data = (testVecData, testConstData), epochs=args['epochs'], batch_size=args['batch_size'], callbacks=callbacks, verbose=args['verbose'])

    #Final evaluation
    eval_metrics = model.evaluate(x=testVecData, y=testConstData, batch_size=args['batch_size'], verbose=args['verbose'])

    #Generate sample neural network output
    predictions = model.predict_on_batch(testVecData)
    predictions = np.reshape(predictions, args['test_points'])
    testErrors = np.absolute(predictions - testConstData)
    maxError = np.amax(testErrors)
    minError = np.amin(testErrors)

    maxIndex = np.argmax(testErrors)
    minIndex = np.argmin(testErrors)

    #Calculate the mode or thereabouts
    testErrors = np.sort(testErrors)
    median_testErrors = testErrors[int(args['test_points']/2)]

    #Print interesting error information
    if args['verbose'] >= 1:
        print('Max error: ', maxError)
        print('Min error: ', minError)
        print('Max Location: ', testVecData[maxIndex, :])
        print('Min Location: ', testVecData[minIndex, :])
        print('Max Location Constant: ', testConstData[maxIndex])
        print('Min Location Constant: ', testConstData[minIndex])
        print('Median of errors: ', median_testErrors)
        print('Standard deviation: ', np.std(predictions))

    #File name setups
    errorFileName = os.path.join('..', 'Errors', 'errors_' + str(n) + '_' + str(m) + '_' + str(args['train_points']) + '_' + str(args['train_points_version']) + '.txt')
    errorFile = open(errorFileName, 'a+')

    #Log error 
    errorFile.write(str(eval_metrics[1]) + ' ' + str(np.std(predictions)) + ' ' + str(median_testErrors) + ' ' + str(args) + ' ' + str(datetime.datetime.now()) + '\n')
    errorFile.close()

    #Send notification that training has terminated
    if args['send_mail']:
        msg = MIMEText('Max error: ' + str(maxError) + '\nMin error: ' + str(minError) + '\nMax Location: ' + str(testVecData[maxIndex, :]) + '\nMin Location: ' + str(testVecData[minIndex, :]) + '\nAverage Error: ' + str(eval_metrics[1]) + '\nMedian of Errors: ' + str(median_testErrors) + '\nStandard Deviation: ' + str(np.std(predictions)) + '\n')
        msg['Subject'] = 'Finished testing on vecs_' + str(n) + '_' + str(m) + '_' + str(args['train_points']) + '.txt.'
        msg['From'] = args['send_mail'][0]
        msg['To'] = args['send_mail'][1]
        
        s = smtplib.SMTP('smtp-relay.tamu.edu')
        s.send_message(msg)
        s.quit()

    #Plot the loss history of the model
    if args['plot_error']:
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('Model Loss over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Values')
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
        plt.savefig(os.path.join('..', 'Errors', args['plot_error']))
    
    #Save the entire history of the model as dictionary
    if args['save_history']:
        with open(os.path.join('..', 'Histories', args['save_history']), 'w') as f:
            f.write(str(model_history.history))

    #Save architecture of model to json file
    if args['save_architecture']:
        model_str = model.to_json()
        with open(os.path.join('..', 'Architectures', args['save_architecture']), 'w') as f:
            f.write(model_str)

    return model
