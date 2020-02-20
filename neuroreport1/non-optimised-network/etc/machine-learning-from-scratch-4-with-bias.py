from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Perceptron:

    def __init__(self, input_len, learn_rate, weights=None):
        """
        The contructor
        intializes the nessary fields
        :param input_len:  the number of input nodes
        :param learn_rate: the reduce multiplier to eliminate <<jumping>> , optimal at 0.1
        :param weights:    the initial weights , if not given , a 0.5 weight is given for every input node
        """
        self.learn_rate = learn_rate
        self.input_len = input_len
        if weights is None:
            self.weights = np.random.random(input_len+1)#+bias
        else:
            self.weights = weights

    threesold = 0.5

    @staticmethod
    def unit_step_function(x):
        """
        This is the activation function , is a plain binary activation function with threesold=0.5
        (defined as static)
        :param x: the weighted sum
        :return: 1 if weighted sum > threesold
        """
        if (x > Perceptron.threesold):
            return 1
        return 0
    @staticmethod
    def normalise_input(in_data):
        return np.append(in_data,np.array([1])) #plus bias input , always at 1

    def __call__(self, in_data):
        """
        calculates the output for a given input
        :param in_data:  the input
        :return: 1 if over threesold , 0 otherwise
        """
        in_data=Perceptron.normalise_input(in_data)
        weighted_input = self.weights * in_data
        weighted_sum = weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)

    def adjust(self, in_data, calculated_result, target_result):
        """
        calculates and adjusts the Perceptron
        :note this mutates the object , so be careful
        :param in_data:  the data
        :param calculated_result: the result of __call__
        :param target_result:  the result of target function
        :return: the error (used to plot the error maybe)
        """
        in_data=Perceptron.normalise_input(in_data)
        error = target_result - calculated_result
        for i in range(len(in_data)):
            correction = error * in_data[i] * self.learn_rate
            self.weights[i] += correction
        return error


def above_line(point, line_func):
    """
    returns true if a given point is above our target function
    :param point: the point to examine
    :param line_func: the target function
    :return: true if the point is above line_func
    """
    x, y = point
    if y > line_func(x):
        return 1
    else:
        return 0


trainData=20
totalData=100
points = np.random.randint(1, totalData, (100, 2)) #create 100 random points in shape 100,2 (100 rows 2 collumns)
p = Perceptron(2, 0.1)                       #create perceptron with 2 inputs and

# unzip the  points
n, k = zip(*points)


def lin1(x):
    """
    our target function(sinple AF)
    :param x: the x
    :return: x+4
    """
    return x + 4

"""
Here we train our perceptron to start mimicking our lin1 target function
We using 100 epochs as you can see at range(...)

"""
for i in range(100):
    for point in points[:trainData]:
        p.adjust(in_data=point, calculated_result=p(point), target_result=above_line(point, lin1))

# line points , (we will plot our target function here)
x = np.linspace(1, 100)
y = [lin1(n) for n in x] #we map our points from x to x+4


# eval points
suc, fail = [], []
cnt=Counter()
for point in range(trainData,len(points)):
    if (p(points[point]) == above_line(points[point], lin1)): #if our perceptron correctly identifies a point as above line
        suc.append(points[point])                              #is a success
        cnt['Success']+=1
    else:
        fail.append(points[point])                              #else IT SUC
        cnt['Fail'] += 1
print(cnt.most_common()[0][1]/(totalData-trainData))

sucx, sucy = zip(*suc)
if(len(fail)):
    failx, faily = zip(*fail)
    plt.scatter(failx, faily, c='r')

# plt.scatter(n,k,c='b')
plt.scatter(sucx, sucy, c='g')

plt.plot(x, y)
plt.show()