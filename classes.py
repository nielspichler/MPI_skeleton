import time
import numpy as np
import pandas as pd

class Object1:
    def __init__(self):
        self.name = "Object1"
        self.value = 1

    def set_value(self, value):
        self.value = value

class Object2:
    def __init__(self, Object1):
        self.name = "Object2"
        self.value = 2
        self.object1 = Object1

    def compute_value(self, f):
        self.value = self.object1.value + 1 + f
        return self.value

class Object3:
    def __init__(self, object2, rank = None):

        self.rank = rank

        if rank is not None:
            self.name = "Object3_%d"%rank
        else:
            self.name = "Object3"
        self.value = 3
        self.object2 = object2

    def do_computation(self, array1, array2, array3, comm, my_rank):

        res = []

        iteration = 0
        save_rate = 30 # save every save_rate %
        total_samples = len(array1) * len(array2) * len(array3) # * len(d_I_max_range)
        save_rate_increment = save_rate * total_samples // 100

        params_snd = np.empty([total_samples, 6], dtype='f')
        iteration_counter = 0

        if my_rank == 0:
            for el1 in array1:
                for el2 in array2:
                    for el3 in array3:

                        wait_time = float(np.random.rand()) # simulate computation time randon between 0 and 1 seconds

                        params_snd[iteration_counter, :] = np.array([self.rank, iteration_counter] + [el1, el2, el3, wait_time])

                        iteration_counter += 1

        params = np.empty([total_samples//comm.Get_size(), 6], dtype='f')
        comm.Scatter(params_snd, params, root=0) # we use the big S scatter to send the arrays to the other processes

        print('MPI rank: ', my_rank, 'iteration from %d to %d' % (params[0,1], params[-1,1]), flush=True)

        local_total_samples = len(params)
        local_save_rate_increment = save_rate * local_total_samples // 100

        for param in params :

            self.object2.object1.set_value(np.sum(param))

            local_res = []
            for i in range(8):
                local_res.append(self.object2.compute_value(param[5]))

            res.append([self.rank] + local_res + param.tolist())

            # save dataset
            iteration += 1
            if iteration % save_rate_increment == 0:
                df = pd.DataFrame(res)

                if iteration == local_save_rate_increment:  # first save
                    df.to_csv(path_or_buf='./Datasets/%s_%d.csv' %(self.name, my_rank), index=False)
                    print('%sPart of dataset saved as %s_%d.csv%s' % ('\x1b[5;30;44m',
                                                                   self.name,my_rank,
                                                                   '\x1b[0m'))
                else:  # append
                    df.to_csv(path_or_buf='./Datasets/%s_%d.csv' %(self.name, my_rank), index=False, mode='a', header=False)
                    print('%sPart of dataset saved as %s_%d.csv%s' % ('\x1b[5;30;44m',
                                                                   self.name, my_rank,
                                                                   '\x1b[0m'))
                # reset data
                res.clear() # we clear the list so only the new stuff gets saved

        if res not in [[], None]:  # save the last part of the dataset

            df = pd.DataFrame(res)
            df.to_csv(path_or_buf='./Datasets/%s_%d.csv' %(self.name, my_rank), index=False, mode='a', header=False)

            print('Dataset saved as %s.csv' % self.name)

        res = pd.read_csv('./Datasets/%s_%d.csv' %(self.name, my_rank), header=None).values.tolist()

        res = comm.gather(res, root=0)

        if my_rank == 0:
            df = pd.DataFrame(res)
            df.to_csv(path_or_buf='./Datasets/%s_full.csv' % self.name, index=False, header=False)

            print('Dataset saved as %s.csv' % self.name)