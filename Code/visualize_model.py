import MPCModel as mpc
import ast
from matplotlib import pyplot as plt
import os
import numpy as np
import sys

AUGMENT = 20  

def print_weights(model, filePrefix):
    plt.gray()
    
    filePref = filePrefix + '_wlayer'
    count = 0

    B = np.ones((AUGMENT, 1))

    for x in model.layers:
        A = x.get_weights()[0]

        if len(A.shape) == 1:
            A = np.reshape(A, (1, A.shape[0]))

        if A.shape[0] <= AUGMENT or A.shape[1] <= AUGMENT:
            
            if A.shape[0] <= AUGMENT:
                A = np.kron(A, B)

            else:
                A = np.kron(A, B.T)

        A = (A - A.min()) / (A.max() - A.min()) / 2
        threshold_indices = A < np.std(A) + np.mean(A) 
        A[threshold_indices] = 0
        plt.imshow(A)
        print(A)

        plt.savefig(filePref + str(count) + '.png', dpi=600)
        count += 1
    return

def print_biases(model, filePrefix):
    plt.gray()
    
    filePref = filePrefix + '_wlayer'
    count = 0

    B = np.ones((AUGMENT, 1))

    for x in model.layers:
        A = x.get_weights()[1]

        if len(A.shape) == 1:
            A = np.reshape(A, (1, A.shape[0]))

        if A.shape[0] <= AUGMENT or A.shape[1] <= AUGMENT:
            
            if A.shape[0] <= AUGMENT:
                A = np.kron(A, B)

            else:
                A = np.kron(A, B.T)

        A = (A - A.min()) / (A.max() - A.min()) / 2
        threshold_indices = A < np.std(A) + np.mean(A) 
        A[threshold_indices] = 0
        plt.imshow(A)
        print(A)

        plt.savefig(filePref + str(count) + '.png', dpi=600)
        count += 1
    return

if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    args = ast.literal_eval(f.read())
    model = mpc.load_model(args)
    print_weights(model, sys.argv[2])
    #print_biases(args)
