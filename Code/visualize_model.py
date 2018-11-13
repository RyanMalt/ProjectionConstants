import MPCModel as mpc
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import model_from_json

import argparse, ast, os, sys


def print_matrix(A, augment, pixel_augment, std_threshold, fileName, dpi):
    #Prevent weird numpy shape errors
    if len(A.shape) == 1:
        A = np.reshape(A, (1, A.shape[0]))
    
    #Check for need to augment then augment in the correct way
    if A.shape[0] <= pixel_augment or A.shape[1] <= pixel_augment:
    
        if A.shape[0] <= pixel_augment:
            A = np.kron(A, augment)

        else:
            A = np.kron(A, augment.T)
    
    #Normalize between 0 and 1 for grayscale
    A = (A - A.min()) / (A.max() - A.min()) / 2

    #Threshold the matrix based on standard deviations
    if std_threshold != None:
        threshold_indices = A < std_threshold*np.std(A) + np.mean(A)
        A[threshold_indices] = 0
    
    #Plot it
    plt.imshow(A)
    
    #Save it to file
    plt.savefig(fileName, dpi=dpi)
    
    return

def print_model_internals(model, printValues=7, filePrefix='', std_threshold=None, pixel_augment=20, dpi=300):
    '''
        enables plotting of matrices internal to the neural network

        Arguments:

        --model: the keras model to be visualized
        --printValues: decimal representation of binary for which values to print (0-5)
            *000 - print nothing (0)
            *001 - print just biases (1)
            *010 - print just weights (2)
            *011 - print weights and biases (3)
            *100 - print activations on specific input (4)
            *101 - print activations on specific input and biases (5)
            *110 - print activations on specific input and weights (6)
            *111 - print everything (7)
        --filePrefix: prefix to put before each image to be saved
        --std_threshold: number of standard deviations to threshold the image
        --pixel_augment: number of pixels to augment low dimensional matrices with
    '''
    plt.gray()
    count = 0

    augment = np.ones((pixel_augment, 1))
    weights = None
    biases = None
    #activations = None

    bin_print = bin(printValues)[2:]

    for x in model.layers:
        #Prevent trying to print concatenation and input layers
        if isinstance(x, keras.layers.core.Dense): 
            #Check for weights    
            if int(bin_print[0]):
                weights = x.get_weights()[0]
                print_matrix(weights, augment, pixel_augment, std_threshold,
                             filePrefix + '_wlayer_' + str(count) + '.png', dpi=dpi)
            #Check for biases
            if int(bin_print[1]):
                biases = x.get_weights()[1]
                print_matrix(biases, augment, pixel_augment, std_threshold,
                             filePrefix + '_blayer_' + str(count) + '.png', dpi=dpi)

            #Check for activations
            #if int(bin_print[2]):
            #    activations = None

            count += 1
    return

if __name__ == '__main__':
    '''Converts commandline arguments to dictionary of attributes

    Dictionary Keys and Default Values:

    arch_weights_file -- takes two arguments first architecture, then weights file (Default: None)
    model_file -- takes .mcfg file which has associated weights and converts it (Default: None)
    file_prefix -- prefix to put before each generated image (Default: '')
    std_threshold -- number of standard deviations to threshold images (Default: None)
    pixel_augment -- number of pixels to augment low matrices with (Default: 20)
    print_values -- decimal value of binary string indicating which to print i.e. 7 (101) print activations and biases (default: 7)
    dpi -- picture quality (default: 300)
    '''
    parser = argparse.ArgumentParser(description='Creates plots based on model internals')
    
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('-a', '--arch_weights_file', nargs=2,
                        default=None, help='takes two arguments a .json for architecture and a .h5 for weights (default: None)')
    config_group.add_argument('-m', '--model_file', 
                        default=None, help='takes .mcfg file which has an associated weight file name in .mcfg and converts it to a model (Default: None)')

    parser.add_argument('-f', '--file_prefix', 
                        default=None, help='prefix to put before each generated image (Default: None)')
    parser.add_argument('-s', '--std_threshold', type=float,
                        default=None, help='number of standard deviations to threshold images (Default: None)')
    parser.add_argument('-p', '--pixel_augment', type=int,
                        default=20, help='number of pixels to augment low dimensional matrices with (Default: 20)')
    parser.add_argument('--print_values', type=int,
                        default=7, help='decimal value of binary string indicating which values to print i.e. 5 (101) prints activations, ignores weights, and prints biases (default: 7)')
    parser.add_argument('-d', '--dpi', type=int,
                        default=300, help='picture quality (default: 300)')
    args = vars(parser.parse_args())
    
    model = None

    if args['arch_weights_file']:
        with open(args['arch_weights_file'][0], 'r') as f:
            s = f.read()
            model = model_from_json(s)

        model.load_weights(args['arch_weights_file'][1])

    if args['model_file']:
        mod_args = None

        with open(args['model_file'], 'r') as f:
            s = f.read()
            mod_args = ast.literal_eval(s)
            model = mpc.load_model(mod_args)

        if mod_args['file_prefix'] != '':
            mod_args['file_prefix'] = mod_args['file_prefix'] + '_'

        model.load_weights(os.path.join('..', 'Weights', mod_args['file_prefix'] + mod_args['save_weights']))

    print_model_internals(model, args['print_values'], args['file_prefix'], args['std_threshold'], args['pixel_augment'], args['dpi'])
