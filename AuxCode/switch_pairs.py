import os
import sys
import ast

#Files are assumed to be in the format
#name_n_m_numPoints.txt
def switch_pairs(vecFileName):
    fileParts = vecFileName.split('_')
    constFileName = '_'.join('const', fileParts[1], fileParts[2], fileParts[3])

    
    return error, params

if __name__ == '__main__':
