import sys
import ast
import MPCModel

print('Starting model  number ' + sys.argv[2])
MPCModel.generate_model(ast.literal_eval(sys.argv[1]))
print('Finished model number ' + sys.argv[2])

