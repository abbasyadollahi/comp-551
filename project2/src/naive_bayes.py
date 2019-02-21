import os
import numpy as np
from features import binary_vector, y_vector

class NaiveBayes:
    def __init__(self):
        self.X = binary_vector()
        self.y = y_vector()

    def predict(self):
        print(sum(self.y))

print('start')
nb = NaiveBayes()
nb.predict()
print('end')
