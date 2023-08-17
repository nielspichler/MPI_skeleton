from classes import Object1, Object2, Object3

import numpy as np

import time

if __name__ == '__main__':


    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    time_start = time.time()

    loop_1_size = 12
    loop_2_size = 10
    loop_3_size = 14

    array1 = np.empty(loop_1_size, dtype = 'f')
    array2 = np.empty(loop_2_size, dtype = 'f')
    array3 = np.empty(loop_3_size, dtype = 'f')

    if my_rank == 0:
        print("MPI rank: ", my_rank, " out of ", size)

        # we only create the objects once, in the master process

        o1 = Object1() # represents the CZM
        o2 = Object2(o1) # represents the OSLS model

        array1 = np.linspace(0,10,loop_1_size, dtype = 'f')
        array2 = np.linspace(0,10,loop_2_size, dtype = 'f')
        array3 = np.linspace(0,10,loop_3_size, dtype = 'f')

    else:
        o2 = None


    #print('MPI rank: ', my_rank, ' broadcasting objects and arrays')
    o2 = comm.bcast(o2, root=0) # we use the small b broadcast to send the objects to the other processes
    #print('MPI rank: ', my_rank, ' o2 broadcasted')


    #print('MPI rank: ', my_rank, ' broadcasting array2 and 3')
    comm.Bcast(array1, root=0) # we use the big B broadcast to send the arrays to the other processes
    comm.Bcast(array2, root=0) # we use the big B broadcast to send the arrays to the other processes
    comm.Bcast(array3, root=0) # we use the big B broadcast to send the arrays to the other processes
    #print('MPI rank: ', my_rank, ' array2 and 3 broadcasted')


    o3 = Object3(o2) # represents the dataset generator

    o3.do_computation(array1, array2, array3, comm, my_rank)

    time_end = time.time()
    print('MPI rank: ', my_rank, ' time elapsed: ', time_end - time_start, ' seconds')