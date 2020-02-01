import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
Demostration of inabillity to solve XOR
"""
class Perceptron:

    def __init__(self,input_len,learn_rate,weights=None):
        """
        The contructor
        intializes the nessary fields
        :param input_len:  the number of input nodes
        :param learn_rate: the reduce multiplier to eliminate <<jumping>> , optimal at 0.1
        :param weights:    the initial weights , if not given , a 0.5 weight is given for every input node
        """
        self.learn_rate=learn_rate
        self.input_len=input_len
        if weights is None:
            self.weights=(np.ones((1,input_len))*0.5)[0]
        else:
            self.weights=weights


    threesold=0.5
    @staticmethod
    def unit_step_function(x):
        """
        This is the activation function , is a plain binary activation function with threesold=0.5
        (defined as static)
        :param x: the weighted sum
        :return: 1 if weighted sum > threesold
        """
        if(x>Perceptron.threesold):
            return 1
        return 0

    def __call__(self,in_data):
        """
        calculate output based on a array of inputs

        :param in_data: the input data
        :return: 1 if the neuron triggers , 0 otherwise
        """
        weighted_input=self.weights*in_data
        weighted_sum=weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)

    def ajust(self,in_data,calculated_result,target_result):
        """
        The learn function
        :param in_data: the input data
        :param calculated_result: the result of __call__
        :param target_result: the result required for no error
        :return: the error
        :note it changes the state of the perceptron , so be careful , this is a mutable object
        """
        error = target_result-calculated_result
        for i in range(len(in_data)):
            correction = error*in_data[i]*self.learn_rate
            self.weights[i]+=correction
        return error

#create a pandas dataframe for having Epoch vs Error data (to plot it)
df = pd.DataFrame({"Epoch" : [0],
                  "Errors" : [1]})


per = Perceptron(2,0.5)
epochs=100
data_out=0

#this is the xor function and our target(it is used in adjust function l8r)
def target_result(x,y):
    return x==y==1

for e in range(epochs):
    for i in range(2):
        for j in range(2):
            data_in=(i,j)
            data_out=per(data_in)
            print(data_in,data_out)
            err=per.ajust(data_in, data_out, target_result(i, j))
            print("epoch "+str(e)+" Error is "+str(err))
            if(e%5==0):
                df=df.append(pd.DataFrame({"Epoch": [e],
                                    "Errors": [err]}))
    print("-------------------------")


#Plot error vs epochs
plt.plot(df["Epoch"], df["Errors"])
plt.title("Simple Line Plot")
plt.show()