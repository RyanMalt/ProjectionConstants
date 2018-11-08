import argparse
import numpy as np
import os
import ast

#NOTE: Still in the process of adding tensorboard support.  everything is broken
def getArgs():
    '''Converts commandline arguments to dictionary of attributes
    
    Dictionary Keys and Default Values:

    activation_function -- function to be used in NN activations (default: 'relu')
    epochs -- number of times to iterate over training data (default: 100)
    batch_size -- number of data points in each mini-batch (default: 1024)
    input_size -- list containing the dimension of R and the dimension of the subspace (default: [4, 2])
    network_layers -- list containing the widths of each hidden layer and the output layer (default: 100 100 50 1)
    skip_connections -- boolean specifying whether to use skip connections or not
    learning_rate -- pretty self explanatory (default: .001)
    optimizer -- name of the keras optimizer to use (default: adam)
    loss_function -- specific function to measure loss on data points (default: 'mse' i.e. mean-squared)
    metric -- metric for measuring the performance of the NN (default: 'mae' i.e. mean-absolute')
    
    train_points -- specifies which data files to use for training data (default: 200000)
    train_points_version -- specifies the ending values of the dataset to be used (default: 'a')
    test_points -- specifies which data files to use for validation (default: 30000)
    test_points_version -- specifies the ending values of the dataset to be used (default: 'a')
    
    augment_lewicki -- (experimental) boolean for applying normalization of v/(1-2v) to data (default: False)
    augment_division -- augments input to the network with 1/x for each value (default: False)
    augment_zeros -- augment input to the network with columns of zeros (default: False)
    random -- boolean for randomizing constants relative to input vectors (default: False)
    
    activation_regularization -- specifies how much to regularize the weights (default: 0)
    bias_regularization -- specifies how much to regularize the biases (default: 0)
    
    file_prefix -- prefix to add to all files when outputting information (default: '')
    plot_error -- file name of where to plot the error history of the model (default: error.png)
    save_weights -- save the weights of the model after training (default: weights.h5)
    save_architecture -- save the architecture of the model after training (default: arch.json)
    save_history -- saves the entire history of the model during training (default: history.txt)

    reproducible -- boolean to specify the randomness seed for reproducible results (default: False)
    verbose -- integer storing the verbosity level to use (default: 0)
    information -- boolean which says to print out default commandline values and exit (default: False)
    send_mail -- send email from user at Arg1 to user at Arg2 (default: No email)
    
    config_file -- use configuration file holding parameters of program instead of commandline (default: None)
    batch_file -- file containing one dictionary of attributes on each line designed to be run in batches
    tensorboard_file -- use tensorboard configuration file to output tensorboard logs
    '''
    parser = argparse.ArgumentParser(description='Trains feedforward neural network')
    parser.add_argument('-a', '--activation_function', choices=['relu', 'softplus'],  
                    default='relu', help='activation functions for each non-output layer (default: relu)')
    parser.add_argument('-e', '--epochs', type=int,  
                    default=100, help='number of training rounds (default: 100)')
    parser.add_argument('-s', '--batch_size', type=int,  
                    default=1024, help='number of training points in each mini-batch (default: 256)')
    parser.add_argument('--input_size', nargs=2, type=int,
                    default=[4, 2], help='dimension of R and subspace (default: 4 2)')
    parser.add_argument('-n', '--network_layers', nargs='*', type=int,
                    default=[100, 100, 50, 1], help='widths of each hidden layer and the output layer (default: 100 100 1)')
    parser.add_argument('-l', '--learning_rate', type=float,  
                    default=.001, help='learning rate for optimizer (default: .001)')
    parser.add_argument('-o', '--optimizer', choices=['adam', 'rmsprop', 'sgd'],  
                    default='adam', help='optimizer for neural network (default: adam)')
    parser.add_argument('-f', '--loss_function', choices=['mse', 'mae', 'mape', 'msle', 'kld'],  
                    default='mae', help='loss function for neural network (default is mean-absolute)')
    parser.add_argument('-m', '--metric', choices=['mse', 'mae', 'mape', 'msle'],  
                    default='mae', help='metric for measuring performance (default is mean-absolute)')
    parser.add_argument('-d', '--train_points', type=int,  
                    default=200000, help='specifies which data files to use for training data (default: 200000)')
    parser.add_argument('--train_points_version',  
                    default='a', help='specifies which version of data file to use (default: "a")')
    parser.add_argument('-t', '--test_points', type=int,  
                    default=30000, help='number of test points NOT training points (default: 30000)')
    parser.add_argument('--test_points_version',  
                    default='a', help='specifies which version of testing data file to use (default: "a")')
    parser.add_argument('--augment_lewicki', action='store_true',
                    help='augment input data with lewicki normalization of v/1-2v (default: false)')
    parser.add_argument('--augment_division', action='store_true',
                    help='augment input data with 1/x for each value (default: false)')
    parser.add_argument('--augment_zeros', action='store_true',
                    help='augment input data with columns of zeros (default: false)')
    parser.add_argument('--random', action='store_true',
                    help='randomly shuffle input data instead of using properly labeled data (default: false)')
    parser.add_argument('--early_stopping', nargs=2, type=float, default=None,
                    help='allows for early stopping and takes two arguments patience and min_delta (default: 10 0 when called)')
    parser.add_argument('-r', '--activation_regularization',   type=float,
                    default=0, help='regularization constant for weights (default: 0)')
    parser.add_argument('-b', '--bias_regularization',   type=float,
                    default=0, help='regularization constant for biases (default: 0)')
    
    parser.add_argument('--file_prefix', default='',
                    help='prefix to attach to all file names for output information (default: None)')
    parser.add_argument('-p', '--plot_error', nargs='?', const='error.png', default=None, 
                    help='plots the error and saves the .png file to given file name (default: error.png)')
    parser.add_argument('--save_weights', nargs='?', const='weights.hdf5', default=None,
                    help='saves weights of model after training for reuse (default: weights.h5)')
    parser.add_argument('--save_architecture', nargs='?', const='arch.json', default=None,
                    help='saves architecture of model after training for reuse (default: arch.json)')
    parser.add_argument('--save_history', nargs='?', const='history.txt', default=None,
                    help='saves history of the model while training (default: history.txt)')

    parser.add_argument('--reproducible', action='store_true',
                    help='sets rng to be the same (use to check for actual changes in performance)')
    parser.add_argument('-v', '--verbose', action='count',
                    default=0, help='include extended debugging output')
    parser.add_argument('-i', '--information', action='store_true',
                    help='print all default arguments and exit')
    parser.add_argument('--send_mail', nargs='?', default=None, const=['ryanm@math.tamu.edu', 'ryan_malthaner@tamu.edu'],
                    help='send email notification from Arg1 to Arg2 once job is completed')
    parser.add_argument('--tensorboard_file', default=None,
                    help='tensorboard file containing dictionary of attributes that go directly into tensorboard log function')
    
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('--config_file', 
                    default=None, help='use configuration file (dictionary format) instead of commandline')
    config_group.add_argument('--batch_file', 
                    help='specifies batch file to use for computation')

    network_group = parser.add_mutually_exclusive_group()
    network_group.add_argument('--skip_connections', action='store_true',
                    help='specifies that skip connections are to be used for the network (default: False)')

    args = parser.parse_args()

    return vars(args)

def preprocessArgs(args):
    '''Do basic verbosity and configuration file processing'''
    if args['config_file']:
        with open(args['config_file'], 'r') as f:
            s = f.read()
            args = ast.literal_eval(s)
    
    if args['verbose'] >= 1:
        print(args)

    if args['information']:
        print(args)
        print('Exiting...')
        exit()
    
    return args

#msgType: 1 ver
def printVMessage(msgs, vThreshold, verbosity):
    if vThreshold <= verbosity:
        for x in msgs:
            print(x)
    return

#grab raw data from file
def getVecData(n, m, numPoints, version="a"):
    '''Gets the raw vector data from VecData file
    
    Returns:
    vecData -- numPoints x n*m matrix
    '''
    vecFileName = os.path.join('..', 'VecData', 'vecs_' + str(n) + '_' + str(m) + '_' + str(numPoints) + '_' + version + '.txt')
    vecData = np.loadtxt(vecFileName)
    return vecData

#grab raw data from file
def getConstData(n, m, numPoints, version="a"):
    '''Gets the raw constant data from ConstData file
    
    Returns:
    constData -- numPoints x 1 vector
    '''
    constFileName = os.path.join('..', 'ConstData','const_' + str(n) + '_' + str(m) + '_' + str(numPoints) + '_' + version + '.txt')
    constData = np.loadtxt(constFileName)
    return constData

#if a value is less than 1/2, ignore it
