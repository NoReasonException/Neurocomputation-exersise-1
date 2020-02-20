import numpy as np


def initMatrixes(input_len, output_len):
    weights = np.random.randint(0, 1, (output_len, input_len))  # define
    weights = [np.random.binomial(100, 0.5, len(x)) for x in weights]
    print(np.array(weights)/100)

initMatrixes(2,3)