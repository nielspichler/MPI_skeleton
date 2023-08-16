from classes import Object1, Object2, Object3

import numpy as np

if __name__ == '__main__':

    '''
    The idea of First attempt is to separate the job of the first loop in o3.do_computation()
    among the mpi processes. The idea is to have each process do a part of the job, and then
    gather the results in the end.
    '''

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    top_loop_size = 12
    lower_loop_size = 10

    array1_snd = None

    if my_rank == 0:
        print("MPI rank: ", my_rank, " out of ", size)

        # we only create the objects once, in the master process

        o1 = Object1() # represents the CZM
        o2 = Object2(o1) # represents the OSLS model

        array1_snd = np.linspace(0,10,12, dtype = 'f')
        array2 = np.linspace(0,10,10, dtype = 'f')
        array3 = np.linspace(0,10,10, dtype = 'f')

    else:
        o2 = None

    # we create the arrays in all processes
    array1_rcv = np.empty(top_loop_size//size, dtype = 'f')
    array2 = np.empty(lower_loop_size, dtype = 'f')
    array3 = np.empty(lower_loop_size, dtype = 'f')


    print('MPI rank: ', my_rank, ' broadcasting objects and arrays')
    o2 = comm.bcast(o2, root=0) # we use the small b broadcast to send the objects to the other processes
    print('MPI rank: ', my_rank, ' o2 broadcasted')

    print('MPI rank: ', my_rank, ' scattering array1')
    comm.Scatter(array1_snd, array1_rcv, root=0) # we use the big S scatter to send the arrays to the other processes
    print('MPI rank: ', my_rank, ' array1 scattered')

    print('MPI rank: ', my_rank, ' broadcasting array2 and 3')
    comm.Bcast(array2, root=0) # we use the big B broadcast to send the arrays to the other processes
    comm.Bcast(array3, root=0) # we use the big B broadcast to send the arrays to the other processes
    print('MPI rank: ', my_rank, ' array2 and 3 broadcasted')


    o3 = Object3(o2, rank=my_rank) # represents the dataset generator

    o3.do_computation(array1_rcv, array2, array3)