from matplotlib import pyplot as plt
import sys
from ast import literal_eval

def getErrorList(fileName):

    errors = []

    with open(fileName, 'r') as file:
        for line in file:
            mean = float(line.split(' ')[0])
            
            mcfg = literal_eval('{' + line.split('{')[1].split('}')[0] + '}')

            #Checks for random mcfg or any kind of regularization
            if not mcfg['random'] and not mcfg['activation_regularization'] > 0 and not mcfg['bias_regularization'] > 0:
                #Checks to see if mcfg has kernel regularization in it
                #If it doesn't, then we know that mcfg['kernel_regularization'] == 0
                #If it does, then we fall into the or and we have to check
                #whether or not the kernel regularizer was used
                #If it was, then we can pass through.  if it wasn't, then we 
                #have a valid data point.  Holy smokes.
                if not 'kernel_regularization' in mcfg or not mcfg['kernel_regularization'] > 0:
                    if 'skip_connections' in mcfg:
                        errors.append((mcfg['network_layers'], mcfg['learning_rate'], mcfg['skip_connections'], mean))
                    else:
                        errors.append((mcfg['network_layers'], mcfg['learning_rate'], False, mean))
                
    return errors

if __name__ == '__main__':
    errors = getErrorList(sys.argv[1])



