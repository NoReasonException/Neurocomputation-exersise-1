from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class PerceptronsLayer:

    def __init__(self, input_len,neurons_len, learn_rate):
        """
        The contructor
        intializes the nessary fields
        :param input_len:  the number of input nodes
        :param learn_rate: the reduce multiplier to eliminate <<jumping>> , optimal at 0.1
        :param weights:    the initial weights , if not given , a 0.5 weight is given for every input node
        """
        self.learn_rate = learn_rate
        self.neurons_len=neurons_len
        self.input_len = input_len
        self.weights = np.random.randint(0, 10, (neurons_len, input_len+1)) / 10#+1 bias
        #self.weights =np.array([[0.2,0.2,0.2]])

    threesold = 0.3

    @staticmethod
    def unit_step_function(x):
        """
        This is the activation function , is a plain binary activation function with threesold=0.5
        (defined as static)
        :param x: the weighted sum
        :return: 1 if weighted sum > threesold
        """
        if (x > PerceptronsLayer.threesold):
            return 1
        return 0
    @staticmethod
    def normalise_input_per_input(in_data):
        return np.append(in_data,np.array([1])) #plus bias input , always at 1
    @staticmethod
    def biasmatrix(matrix, x, y):
        a = np.ones((x, y + 1))
        for i in range(x):
            for j in range(y):
                a[i][j] = matrix[i][j]
        return a

    def __call__(self, in_data):
        """
        a set of input data
        returns a list of all neurons answers
                          /-----\
                0--------|       \----0
                 .       |       /
                  .      .\-----/
                   .   .
                    . .   /-----\
                    .  .  |      \----0
                  .       |      /
                0-------- \-----/



        """
        results=[]
        in_data=PerceptronsLayer.normalise_input_per_input(in_data) #add bias
        for everyNeuron in range(self.neurons_len):
            weights= self.weights[everyNeuron]
            assert len(in_data) == len(weights)
            weighted_input = weights*in_data
            weighted_sum = weighted_input.sum()
            results.append(PerceptronsLayer.unit_step_function(weighted_sum))
        return results

    def adjust(self, in_data, calculated_result, target_result):
        """
        accepts a pair of input data and for each neuron a desired output , with the calculated ones
        :param in_data:             the input data
        :param calculated_result:   a list with each neurons output data
        :param target_result:       a list with each neurons target data
        :return:                    the error table for this iteration
        """
        in_data = PerceptronsLayer.normalise_input_per_input(in_data)#add bias
        errortbl = target_result - calculated_result
        for everyCalculatedResult in range(len(calculated_result)):
            assert len(calculated_result) == len(target_result)
            assert len(self.weights[everyCalculatedResult])==len(in_data)
            assert len(self.weights)==self.neurons_len
            correction = in_data*errortbl[everyCalculatedResult]*self.learn_rate
            self.weights[everyCalculatedResult]=self.weights[everyCalculatedResult]+correction
        return errortbl


def genEpochInput():
    return np.array([[0,0],[0,1],[1,0],[1,1]])
def genEpochOutput():
    return np.array([[0,0,1],[1,0,1],[1,0,1],[1,1,1]])


p = PerceptronsLayer(2,3,0.1)

epochs = 100





for i in range(epochs):
    inp = genEpochInput()
    #inp=PerceptronsLayer.biasmatrix(inp,inp.shape[0],inp.shape[1])
    out = genEpochOutput()
    adjust=0
    for eachDataset in range(len(inp)):
        result = p(inp[eachDataset])
        adjust = p.adjust(inp[eachDataset], result, out[eachDataset])


print(p(np.array([0,0])))
print(p(np.array([0,1])))
print(p(np.array([1,0])))
print(p(np.array([1,1])))