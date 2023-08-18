from classes import Object1, Object2

import numpy as np
import pandas as pd

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

    name = "Object3_%d" % my_rank

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


    # object 3 is deleted and the method goes here

    res = []

    iteration = 0
    save_rate = 30  # save every save_rate %
    total_samples = len(array1) * len(array2) * len(array3)  # * len(d_I_max_range)
    save_rate_increment = save_rate * total_samples // 100

    params_snd = np.empty([total_samples, 6], dtype='f')
    iteration_counter = 0

    if my_rank == 0:
        for el1 in array1:
            for el2 in array2:
                for el3 in array3:
                    wait_time = float(np.random.rand()) * 0.0001  # simulate computation time randon between 0 and 1 seconds

                    params_snd[iteration_counter, :] = np.array(
                        [my_rank, iteration_counter] + [el1, el2, el3, wait_time])

                    iteration_counter += 1


    params = np.empty([total_samples // comm.Get_size(), 6], dtype='f')
    comm.Scatter(params_snd, params, root=0)  # we use the big S scatter to send the arrays to the other processes

    print('MPI rank: ', my_rank, 'iteration from %d to %d' % (params[0, 1], params[-1, 1]), flush=True)

    local_total_samples = len(params)
    local_save_rate_increment = save_rate * local_total_samples // 100

    for param in params:

        o2.object1.set_value(np.sum(param))

        local_res = []
        for i in range(8):
            local_res.append(o2.compute_value(param[5]))
            #time.sleep(param[5] / 8)

        res.append([my_rank] + local_res + param.tolist())

        # save dataset
        iteration += 1
        if iteration % save_rate_increment == 0:
            df = pd.DataFrame(res)

            if iteration == local_save_rate_increment:  # first save
                df.to_csv(path_or_buf='./Datasets/%s.csv' % (name), index=False)
                print('%sPart of dataset saved as %s.csv%s' % ('\x1b[5;30;44m',
                                                                  name,
                                                                  '\x1b[0m'))
            else:  # append
                df.to_csv(path_or_buf='./Datasets/%s.csv' % (name), index=False, mode='a',
                          header=False)
                print('%sPart of dataset saved as %s.csv%s' % ('\x1b[5;30;44m',
                                                                  name,
                                                                  '\x1b[0m'))
            # reset data
            res.clear()  # we clear the list so only the new stuff gets saved

    if res not in [[], None]:  # save the last part of the dataset

        df = pd.DataFrame(res)
        df.to_csv(path_or_buf='./Datasets/%s.csv' % (name), index=False, mode='a', header=False)

        print('Dataset saved as %s.csv' % name)

    res = pd.read_csv('./Datasets/%s.csv' % (name), header=None).values

    res = comm.gather(res, root=0)

    if my_rank == 0:

        res = np.concatenate(res, axis=0)

        df = pd.DataFrame(res)
        df.to_csv(path_or_buf='./Datasets/FULL.csv', index=False, header=False)

        print('Dataset saved as FULL.csv')


    time_end = time.time()
    print('MPI rank: ', my_rank, ' time elapsed: ', time_end - time_start, ' seconds')