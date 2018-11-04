import numpy as np
import sys

#Assumes file of the form str_n_m_numPoints_version.ending
def inf_normalize(fileName):
    A = np.loadtxt(fileName)
    
    parts = fileName.split('_')
    n = int(parts[1])
    m = int(parts[2])

    for i in range(m):
        A[:, i*n:(i + 1)*n] = (A[:, i*n:(i + 1)*n].T/np.max(np.abs(A[:, i*n:(i + 1)*n]), axis=1)).T
        
    newFile = '_'.join([parts[0], parts[1], parts[2], parts[3], 'inf' + parts[4]])
    np.savetxt(newFile, A, fmt='%.10f')

    return

if __name__ == '__main__':
    inf_normalize(sys.argv[1])
