from options import getArgs, preprocessArgs 
import ast
import sys
import subprocess

args = getArgs()
args = preprocessArgs(args)

#NOTE: Most likely in the future, this will need some kind of thread/process safe
#way of giving updates on what model is currently being processed and how far along
#it is
if args['batch_file']:
    
    #open file in safe way
    with open(args['batch_file']) as batch_file:
        batch_text = batch_file.readlines()
        batches = ast.literal_eval(batch_text[0])
        
        #Range over batches, then range over each individual batch
        for i in range(int(batches[0] / batches[1])):
            procs = []
            for j in range(batches[1]):
                proc = subprocess.Popen([sys.executable, 'slave.py', batch_text[1 + batches[1] * i + j], str(batches[1] * i + j)])
                procs.append(proc)
            
                #Ensures we don't have an explosion of processes
            for proc in procs:
                proc.wait()
       
        #If it turns out the batch size and the total models aren't evenly divisible
        #then we have to do this
        if int(batches[0] / batches[1]) * batches[1] != batches[0]:
            procs = []
            start = int(batches[0] / batches[1]) * int(batches[1])
            for i in range(int(batches[0]) - start):
                proc = subprocess.Popen([sys.executable, 'slave.py', batch_text[start + i], str(start + i)])
                procs.append(proc)

            for proc in procs:
                proc.wait()
    
elif args['config_file']:
    proc = subprocess.Popen([sys.executable, 'slave.py', str(args), str(0)])
    proc.wait()

#Case for commandline arguments
else:
    proc = subprocess.Popen([sys.executable, 'slave.py', str(args), str(0)])
    proc.wait()


