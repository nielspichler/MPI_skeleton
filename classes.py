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

    def do_computation(self, array1, array2, array3):

        res = []

        iteration = 0
        save_rate = 10 # save every save_rate %
        total_samples = len(array1) * len(array2) * len(array3) # * len(d_I_max_range)
        save_rate_increment = save_rate * total_samples // 100

        for el1 in array1:
            for el2 in array2:
                for el3 in array3:

                    wait_time = float(np.random.rand()) # simulate computation time randon between 0 and 1 seconds

                    #time.sleep(wait_time) # simulate computation time randon between 0 and 1 seconds

                    # roughly thr structure of the computation

                    self.object2.object1.set_value(el1)

                    local_res = []
                    for i in range(8):
                        local_res.append(self.object2.compute_value(wait_time))

                    res.append([self.rank, el1, el2, el3] + local_res)

                    # save dataset
                    iteration += 1
                    if iteration % save_rate_increment == 0:
                        df = pd.DataFrame(res)

                        if iteration == save_rate_increment:  # first save
                            df.to_csv(path_or_buf='./Datasets/%s.csv' % self.name, index=False)
                            print('%sPart of dataset saved as %s.csv%s' % ('\x1b[5;30;44m',
                                                                           self.name,
                                                                           '\x1b[0m'))
                        else:  # append
                            df.to_csv(path_or_buf='./Datasets/%s.csv' % self.name, index=False, mode='a', header=False)
                            print('%sPart of dataset saved as %s.csv%s' % ('\x1b[5;30;44m',
                                                                           self.name,
                                                                           '\x1b[0m'))
                        # reset data
                        data = []

        if res not in [[], None]:  # save the last part of the dataset

            df = pd.DataFrame(res)
            df.to_csv(path_or_buf='./Datasets/%s.csv' % self.name, index=False, mode='a', header=False)

            print('Dataset saved as %s.csv' % self.name)