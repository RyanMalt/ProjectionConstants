import os
import sys
import numpy as np
import random

#NOTE: Due to changes in file version system, this is broken
#Fileterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

    return
#files are assumed to be in the format
#vecs_n_m_numPoints_version.txt 
#newPoints is the number of points per old point
def augment_data(vecFileName, newPoints):
    #Parse the file name for n, m, numPoints
    newPoints = int(newPoints)
    fileParts = vecFileName.split('_')
    constFileName = '_'.join(['const', fileParts[1], fileParts[2], fileParts[3], fileParts[4]])
    
    #Open vec and const files
    with open(os.path.join('..', 'VecData', vecFileName), 'r') as vecFile:
        with open(os.path.join('..', 'ConstData', constFileName), 'r') as constFile:
        
            #Grab n, m, and the new number of points
            newNumPoints = int(fileParts[3].split('.')[0]) * newPoints + int(fileParts[3].split('.')[0])
            n = int(fileParts[1])
            m = int(fileParts[2])

            #Used to generate sign matrices later
            choices = [-1,1]
    
            #Create the new file names for the augmented files
            newVecFileName = '_'.join(['vecs', fileParts[1], fileParts[2], str(newNumPoints), fileParts[4]]) 
            newConstFileName = '_'.join(['const', fileParts[1], fileParts[2], str(newNumPoints), fileParts[4]]) 
            
            #Crack em open
            with open(os.path.join('..', 'VecData', newVecFileName), 'w+') as newVecFile:
                with open(os.path.join('..', 'ConstData', newConstFileName), 'w+') as newConstFile:
                    print('Augmenting data...')
                    count = 0
                    #Grab the vectors and const from each line of the files
                    for vec, const in zip(vecFile, constFile):
                        #Write the old stuff to the new files
                        newVecFile.write(vec)
                        newConstFile.write(const)
                        count += 1
                        if count % 100 == 0:
                            printProgressBar(count, newNumPoints, 'Progress', 'Completed', length=50)

                        #Format the vectors as a proper n x m matrix
                        vec = np.reshape(np.array([float(i) for i in vec.split()]), (n, m), 'F')
                        for i in range(newPoints):
                            #Generate sign matrix of size mxm
                            signs = np.reshape(np.array(random.choices(choices, k=m*m)), (m, m))
                            
                            #Generate weights of size mxm
                            weights = np.random.rand(m, m)
                            weights = signs * weights

                            #Construct new vectors by multiplying nxm by mxm
                            newVecs = np.dot(vec, weights)
                            newVecs = newVecs / np.abs(newVecs).sum(axis=0)[:]
                            newVecs = np.reshape(newVecs, (1, n*m), 'F').flatten()

                            #Write vector to file and const corresponding to it
                            newVecFile.write(' '.join(str(x) for x in newVecs) + '\n')
                            newConstFile.write(const)
                            count += 1

    return
                            
if __name__ == '__main__':
    augment_data(sys.argv[1], sys.argv[2])
