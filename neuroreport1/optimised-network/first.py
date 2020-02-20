import numpy as np
import math

from scipy.stats import truncnorm


class NeuralNetwork:
    def __init__(self,layers:iter):
        self.layers=layers
        pass
    def __call__(self, in_data):
        retval=[]
        for eachLayer in range(len(self.layers)):
            if(eachLayer==0):
                retval.append(self.layers[eachLayer](in_data))
            else:
                retval.append(self.layers[eachLayer](retval[-1]))
        return retval

    def adjust(self, in_data, target_result, calculated_result):
        pass


class PerceptronsMultiLayer:



    @staticmethod
    def binInit(input_len,output_len):
        weights = np.random.randint(0,1, (output_len,input_len)) #define
        weights = [np.random.binomial(100,0.5,len(x)) for x in weights]
        return np.array(weights)/100

    @staticmethod
    def truncNormInit(input_len,output_len):
        def truncated_normal(mean:float=0.0, sd:float=1.0, low:float=0.0, upp:float=10.0):
            return truncnorm(
                (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        weights = np.random.randint(0, 1, (output_len, input_len))  # define
        weights = [truncated_normal(0, 1,(-1/math.sqrt(input_len)),1/math.sqrt(input_len)).rvs(len(x)) for x in weights]
        return np.array(weights)





    def __init__(self,index,input_len,output_len,activation,weights=None):
        self.index=index
        shape=(output_len,input_len)
        if(weights==None):
            self.weights=PerceptronsMultiLayer.binInit(input_len,output_len)
        else:
            self.weights=np.array(weights)

        self.activation=activation
        assert self.weights.shape == shape


    def __call__(self, prev_layer_out):

        results=prev_layer_out*self.weights
        retval = [everyResult.sum() for everyResult in results]
        print("layer" + str(self.index) + " input" + str(prev_layer_out) +"out "+str(retval))
        return retval





def linear(sum):
    return sum
def sigmoid(sum):
    return 1/(1+math.pow(math.e,-sum))
def reLU(sum):
    return max(0,sum)

nIn = PerceptronsMultiLayer(0,2,3,linear)
nOut = PerceptronsMultiLayer(1,3,2,linear)

network = [nIn,
           nOut]

wrapper=NeuralNetwork(network)

print(wrapper([1,1]))



