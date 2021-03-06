import numpy as np
import math

from scipy.stats import truncnorm



@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


class NeuralNetwork:
    def __init__(self,layers:iter):
        self.layers=layers
        pass
    def __call__(self, in_data):
        retval=[]
        for eachLayer in range(len(self.layers)):
            if(eachLayer==0):
                retval.append(self.layers[eachLayer](NeuralNetwork.normalisation(in_data)))
            else:
                retval.append(self.layers[eachLayer](NeuralNetwork.normalisation(retval[-1])))
        return retval

    @staticmethod
    def normalisation(in_data):
        return np.append(in_data, np.array([1]))  # plus bias input , always at 1

    def adjust(self, in_data, target_result, calculated_result):
        for everyLayer in range(len(self.layers) - 1, -1, -1):  # traverse the list on reverse
            if (everyLayer == len(self.layers) - 1):  # we are in the last layer
                prev_weights = (self.layers[everyLayer].term_learn(
                    NeuralNetwork.normalisation(calculated_result[everyLayer - 1]),
                    calculated_result[everyLayer], target_result))

            elif (everyLayer == 0):  # we are in the first layer , and the input is the networks input
                prev_weights = (self.layers[everyLayer].mid_learn(NeuralNetwork.normalisation(in_data),
                                                                     self.layers[everyLayer + 1],
                                                                     calculated_result[everyLayer]))

                continue

            else:  # middle train
                prev_weights = (
                    self.layers[everyLayer].mid_learn(NeuralNetwork.normalisation(calculated_result[everyLayer - 1]),
                                                         self.layers[everyLayer + 1],
                                                         calculated_result[everyLayer]))

                continue

        return 1


class PerceptronsMultiLayer:

    def roundWeights(self):
        for i in range(self.output_len):
            self.weights[i] = [np.round(x, 4) for x in self.weights[i]]


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

    def __init__(self,index,input_len,output_len,activation,learning_rate,momentum,weights=None):
        self.index=index
        self.Delta=[]
        self.correction=[]
        self.input_len=input_len
        self.output_len = output_len
        shape=(output_len,input_len+1)
        self.learning_rate=learning_rate
        if(weights is None):
            self.weights=PerceptronsMultiLayer.binInit(input_len+1,output_len)#bias
        else:
            self.weights=np.array(weights)

        self.activation=activation
        assert self.weights.shape == shape

    @staticmethod
    def biased(input:np.array):
        return np.append(input, np.array([1]))

    def __call__(self, prev_layer_out):
        sum=np.dot(self.weights,prev_layer_out.T)
        result=sigmoid(sum)
        return result.T

    def mid_learn(self, i, next_layer, o):
        Wr_p_1 = next_layer.weights
        Dr_p_1 = next_layer.Delta
        i=np.array(i).reshape(1,self.input_len+1)
        o = np.append(o,1).reshape(self.output_len+1, 1)
        self.Delta = np.matmul(Wr_p_1.T,Dr_p_1) * (1.0 - o) * o * self.learning_rate
        correction = i * self.Delta
        self.weights += correction[:-1]
        self.roundWeights()
        pass



    def term_learn(self, i, o, t):
        t=t.reshape(1,self.output_len)
        o=o.reshape(1, self.output_len)
        self.Delta=np.array((t-o))
        self.Delta=self.Delta*(1.0-o)*o
        correction=i*self.Delta.T*self.learning_rate
        self.weights+=correction
        self.roundWeights()

nIn = PerceptronsMultiLayer(0,2,2,sigmoid,0.5,0.6,weights=np.array([[-0.1558, 0.2829, 0.8625], [-0.5060, -0.8644, 0.8350]]))
nOut = PerceptronsMultiLayer(1,2,1,sigmoid,0.5,0.6,weights=np.array([[-0.4304, 0.4812, 0.0365]]))
network = [nIn,
           nOut]

wrapper=NeuralNetwork(network)

data=np.array([[0,0],[0,1],[1,0],[1,1]])
output=np.array([[0],[1],[1],[0]])

epochs=10000
"""
for i in range(epochs):
    for i in range(len(data)):
        mid_out=nIn(data[i])       #layer2 out

        out=nOut(mid_out)            #layer1 out

        nOut.term_learn(mid_out, out, output[i])

        nIn.mid_learn(data[i],nOut,mid_out)
"""
for i in range(1):
    for j in range(1):
        wrapper.adjust(data[j],output[j],wrapper(data[j]))

print(nIn.weights)
print(nOut.weights)

print("------------")
print(wrapper(data[0])[-1])
print(wrapper(data[1])[-1])
print(wrapper(data[2])[-1])
print(wrapper(data[3])[-1])


