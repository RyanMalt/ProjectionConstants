import sys
import ast
import MPCModel
import datetime

print('Starting model number ' + sys.argv[2] + ' at time ' + str(datetime.datetime.now()))
sys.stdout.flush()
MPCModel.generate_model(ast.literal_eval(sys.argv[1]))
print('Finished model number ' + sys.argv[2] + ' at time ' + str(datetime.datetime.now()))
sys.stdout.flush()

