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
