import os
import sys
import ast

def find_min_error(fileName):
    error = 10000
    params = None

    with open(fileName, 'r') as f:
        for line in f:
            line = line.split(' ', 3)
            if float(line[0]) < error:
                error = float(line[0])
                params = line[3].split('}')[0] + '}'
                params = ast.literal_eval(params)

    return error, params

if __name__ == '__main__':
    error, params = find_min_error(sys.argv[1])
    print('Minimum error from ' + sys.argv[1] + ': ' + str(error))
    print('Network architecture: ' + str(params))
