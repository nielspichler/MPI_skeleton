from classes import Object1, Object2, Object3

import numpy as np

if __name__ == '__main__':

    o1 = Object1() # represents the CZM
    o2 = Object2(o1) # represents the OSLS model
    o3 = Object3(o2) # represents the dataset generator

    array1 = np.arange(0, 10, 1)
    array2 = np.arange(0, 10, 1)
    array3 = np.arange(0, 10, 1)

    o3.do_computation(array1, array2, array3)