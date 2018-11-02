import ast
import sys
import argparse
from itertools import chain, product

'''
    Generates batch of network configurations to train on later
    
    first line of batch configuration file should be a dictionary
    holding all values necessary for training a network i.e.
    structured in the format of a model.cfg file

    models_per_batch -- number of processes to be spawned at one time
    network_widths -- list of lists of the form [a, b, delta_x] where a is the
                        starting width for a list, and b is the ending width,
                        while delta_x specifies how to increment from a to b.
                        Eventually, this will be expanded to something of the
                        form [a, a + delta_x, ..., a + i*delta_x, ...,].
                        NOTE: Specifying a = b implies this hyperparameter will
                        be kept constant, even with cross_training
    learning_rates -- list of learning rates with same format as network widths
    activation_regularizations -- list of activation regularizations with same
                                    format as network widths
    bias _regularazations -- list of bias regularizations with same format as 
                                    network widths
    batch_sizes -- list of batch sizes with same format as network widths
    cross_training -- states which hyperparameters should be chosen to consider
                        all possible combinations of values from each of their
                        range of values e.g. if we have [True, True, True,
                        False, False], this would mean that all possible
                        combinations of network widths, learning rates, and 
                        batch sizes are added to the pool of network parameters
                        to be trained over
'''
def gen_list(param_list):
    '''Takes a list of the form [a, b, delta_x] and expands it'''
    
    new_list = []
    num_steps = int((param_list[1] - param_list[0])/param_list[2]) + 1
    for x in range(num_steps):
        new_list.append(param_list[0] + x * param_list[2])

    return new_list

#Takes dictionary with file name of batch config file and output file
def generate_batch(args):
    #Constants for referencing values of cross_train list
    NETWORK_WIDTHS = 0
    LEARNING_RATES = 1
    ACTIVATION_REGS = 2
    BIAS_REGS = 3
    BATCH_SIZES = 4


    batch_config = None
    default_model = None
    with open(args['config_file'], 'r') as f:
        default_model = ast.literal_eval(f.readline())
        batch_config = ast.literal_eval(f.readline())

    model_list = []

    num_networkModels = 1
    for layer in batch_config['network_widths']:
        num_networkModels *= int((layer[1] - layer[0])/layer[2] + 1)
    model_list.append(num_networkModels)

    model_list.append(int((batch_config['learning_rates'][1] - batch_config['learning_rates'][0])/batch_config['learning_rates'][2]) + 1)
    model_list.append(int((batch_config['activation_regularizations'][1] - batch_config['activation_regularizations'][0])/batch_config['activation_regularizations'][2]) + 1)
    model_list.append(int((batch_config['bias_regularizations'][1] - batch_config['bias_regularizations'][0])/batch_config['bias_regularizations'][2]) + 1)
    model_list.append(int((batch_config['batch_sizes'][1] - batch_config['batch_sizes'][0])/batch_config['batch_sizes'][2]) + 1)

    totalModelsCT = 1
    totalModelsNoCT = 0
    totalModels = 0

    plotChange = False
    weightChange = False
    archChange = False
    histChange = False

    #Checks to see if there are files that need to be mass generated
    if default_model['plot_error']:
        plotChange = True
    if default_model['save_weights']:
        weightChange = True
    if default_model['save_architecture']:
        archChange = True
    if default_model['save_history']:
        histChange = True

    for i in range(len(batch_config['cross_training'])):
        if batch_config['cross_training'][i]:
            totalModelsCT *= model_list[i]
        else:
            totalModelsNoCT += model_list[i]

    totalModels = totalModelsCT + totalModelsNoCT

    with open(args['output_file'], 'w+') as f:
        f.write(str([totalModels, batch_config['models_per_batch']]) + '\n')
        
        #Generate proper lists of values to iterate over
        network_products = list(product(gen_list(batch_config['network_widths'][0]), gen_list(batch_config['network_widths'][1])))
        for layer in batch_config['network_widths'][2:]:
            network_products = list(product(network_products, gen_list(layer)))
            
            for x in range(len(network_products)):
                network_products[x] = list(chain(network_products[x][0], [network_products[x][1]]))

        x_train_stack = []
        no_x_train_stack = []
        actual_total = 0
        #NOTE: SET NON CROSS TRAIN PARTS OF LISTS TO DEFAULT VALUES
        #Generate stacks for cross training and individual training
        #Goal: Create tuples of the form (n0, n1, ..., nd, lr, ar, br, bs)
        if sum(batch_config['cross_training']) >= 2:
            if batch_config['cross_training'][NETWORK_WIDTHS]:
                x_train_stack.append(network_products)
            else:
                x_train_stack.append(default_model['network_layers'])
                no_x_train_stack.append([network_products, 'network_layers'])

            if batch_config['cross_training'][LEARNING_RATES]:
                x_train_stack.append(gen_list(batch_config['learning_rates']))
            else:
                x_train_stack.append([default_model['learning_rate']])
                no_x_train_stack.append([gen_list(batch_config['learning_rates']), 'learning_rate'])

            if batch_config['cross_training'][ACTIVATION_REGS]:
                x_train_stack.append(gen_list(batch_config['activation_regularizations']))
            else:
                x_train_stack.append([default_model['activation_regularization']])
                no_x_train_stack.append([gen_list(batch_config['activation_regularizations']), 'activation_regularization'])

            if batch_config['cross_training'][BIAS_REGS]:
                x_train_stack.append(gen_list(batch_config['bias_regularizations']))
            else:
                x_train_stack.append([default_model['bias_regularization']])
                no_x_train_stack.append([gen_list(batch_config['bias_regularizations']), 'bias_regularization'])

            if batch_config['cross_training'][BATCH_SIZES]:
                x_train_stack.append(gen_list(batch_config['batch_sizes']))
            else:
                x_train_stack.append([default_model['batch_size']])
                no_x_train_stack.append([gen_list(batch_config['batch_sizes']), 'batch_size']) 

            output_model = default_model.copy()
            for params in product(x_train_stack[0], x_train_stack[1], x_train_stack[2], x_train_stack[3], x_train_stack[4]):
                output_model['network_layers'] = params[NETWORK_WIDTHS]
                output_model['learning_rate'] = params[LEARNING_RATES]
                output_model['activation_regularization'] = params[ACTIVATION_REGS]
                output_model['bias_regularization'] = params[BIAS_REGS]
                output_model['batch_size'] = params[BATCH_SIZES]

                #Modifies file names so they don't overwrite each other
                if plotChange:
                    output_model['plot_error'] = default_model['plot_error'][:-4] + '_' + str(actual_total) + default_model['plot_error'][-4:]
                if weightChange:
                    output_model['save_weights'] = default_model['save_weights'][:-3] + '_' + str(actual_total) + default_model['save_weights'][-3:]
                if archChange:
                    output_model['save_architecture'] = default_model['save_architecture'][:-5] + '_' +  str(actual_total) + default_model['save_architecture'][-5:]
                if histChange:
                    output_model['save_history'] = default_model['save_history'][:-4] + '_' + str(actual_total) + default_model['save_history'][-4:]

                f.write(str(output_model) + '\n')
                actual_total += 1

            #That handles the cross training, now we have to hit everything else
            output_model = default_model.copy()
            for params, param_name in no_x_train_stack:
                if len(params) > 1:
                    for param in params:
                        output_model[param_name] = param
                        if plotChange:
                            output_model['plot_error'] = default_model['plot_error'][:-4] + '_' + str(actual_total) + default_model['plot_error'][-4:]
                        if weightChange:
                            output_model['save_weights'] = default_model['save_weights'][:-3] + '_' +  str(actual_total) + default_model['save_weights'][-3:]
                        if archChange:
                            output_model['save_architecture'] = default_model['save_architecture'][:-5] + '_' + str(actual_total) + default_model['save_architecture'][-5:]
                        if histChange:
                            output_model['save_history'] = default_model['save_history'][:-4] + '_' + str(actual_total) + default_model['save_history'][-4:]

                        f.write(str(output_model) + '\n')
                        actual_total += 1
                    output_model = default_model.copy()

        #If no cross training is required, just output standard dictionaries for each changeable parameter
        else:
            output_model = default_model.copy()
            train_stack = []

            train_stack.append([network_products, 'network_layers'])
            train_stack.append([gen_list(batch_config['learning_rates']), 'learning_rate'])
            train_stack.append([gen_list(batch_config['activation_regularizations']), 'activation_regularization'])
            train_stack.append([gen_list(batch_config['bias_regularizations']), 'bias_regularization'])
            train_stack.append([gen_list(batch_config['batch_sizes']), 'batch_size'])

            for params, param_name in train_stack:
                if len(params) > 1:
                    for param in params:
                        output_model[param_name] = param
                        
                        if plotChange:
                            output_model['plot_error'] = default_model['plot_error'][:-4] + '_' + str(actual_total) + default_model['plot_error'][-4:]
                        if weightChange:
                            output_model['save_weights'] = default_model['save_weights'][:-3] + '_' + str(actual_total) + default_model['save_weights'][-3:]
                        if archChange:
                            output_model['save_architecture'] = default_model['save_architecture'][:-5] + '_' +  str(actual_total) + default_model['save_architecture'][-5:]
                        if histChange:
                            output_model['save_history'] = default_model['save_history'][:-4] + '_' + str(actual_total) + default_model['save_history'][-4:]

                        f.write(str(output_model) + '\n')
                        actual_total += 1
                    output_model = default_model.copy()

        return actual_total 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates batch file for training multiple neural networks concurrently')
    parser.add_argument('-o', '--output_file', default='batch.txt',
                help='Name of the file to store the batch information in')
    parser.add_argument('-f', '--config_file',
                help='File with dictionary of attributes to generate batch file')
    args = vars(parser.parse_args())

    print(generate_batch(args))
    

