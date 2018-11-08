import ast
import sys
from generate_batch import generate_batch

'''
    arg1: starting batch configuration file
    arg2: starting depth
    arg3: ending depth
    arg4: widths at each layer
'''
#NOTE: Make sure to double check dictionary keys
with open(sys.argv[1], 'r') as f:
    default_model = ast.literal_eval(f.readline())
    var_model = dict(default_model)
    bat = ast.literal_eval(f.readline())

    depth_start = int(sys.argv[2]) + 1 #we assume the first batch is already made
    depth_finish = int(sys.argv[3])
    widths = int(sys.argv[4])
    count = 0

    for i in range(depth_start, depth_finish + 1):
        count += 1
        if var_model['plot_error']:
            var_model['plot_error'] = var_model['plot_error'][:-4] + '_d' + str(i) + var_model['plot_error'][-4:]
        if var_model['save_weights']:
            var_model['save_weights'] = var_model['save_weights'][:-3] + '_d' + str(i) + var_model['save_weights'][-3:]
        if var_model['save_architecture']:
            var_model['save_architecture'] = var_model['save_architecture'][:-5] + '_d' + str(i) + var_model['save_architecture'][-5:]
        if var_model['save_history']:
            var_model['save_history'] = var_model['save_history'][:-4] + '_d' + str(i) + var_model['save_history'][-4:]
        
        bat['network_widths'].insert(-2, [widths, widths, widths])
        batch_name = 'depth_' + str(i) + '_' + sys.argv[1]
        
        with open(batch_name, 'w') as d:
            d.write(str(var_model))
            d.write('\n')
            d.write(str(bat))
    
        gen_dict = dict()
        gen_dict['config_file'] = batch_name
        gen_dict['output_file'] = sys.argv[1][:-5] + '_d' + str(i) + '.txt'

        #Generate batch for this setup
        generate_batch(gen_dict)
        
        var_model = dict(default_model)

    print(count)
