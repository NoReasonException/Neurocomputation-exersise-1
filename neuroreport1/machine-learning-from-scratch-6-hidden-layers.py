
import numpy as np


class NeuralNetwork:
    def __init__(self,layers:iter):
        NeuralNetwork.assertAllTrue([isinstance(x, PerceptronsLayer) for x in layers])
        if(NeuralNetwork.layerIntegrityCheck(layers)):
            self.layers=layers

    @staticmethod
    def assertAllTrue(booleanList:iter)->bool:
        for everyCheck in booleanList:
            assert everyCheck,("Neural Network's integrity rules violated")
        return True

    @staticmethod
    def layerIntegrityCheck(layers:iter)->bool:
        assert len(layers)>1
        for i in range(len(layers)-1):
            outputs = layers[i].neurons_len
            inputs = layers[i+1].input_len
            assert inputs==outputs
        return True


    def getNthLayerOutput(self,in_data:iter,n:int)->list:
        """
        receives the output of n-1 layer and calculates the output of nth layer
        :param input: the output of n-1 layer , used as an input
        :param n: the index of the layer we want to calculate
        :return: a list with the outputs on nth layer
        """
        nthLayer=self.layers[n]
        return nthLayer.__call__(in_data)


    def __call__(self, in_data,n:int):
        result=in_data
        for i,everyLayer in enumerate(self.layers):
            if(i==0):
                result=np.vstack([result,self.getNthLayerOutput(in_data,i)])
            else:
                result=np.vstack([result, self.getNthLayerOutput(result[-1], i)])
        return result[1:]
    





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
        #self.weights = np.random.randint(0, 10, (neurons_len, input_len)) / 10
        self.weights =np.array([[1,1],[1,1]])

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
#        in_data=PerceptronsLayer.normalise_input_per_input(in_data) #add bias
        for everyNeuron in range(self.neurons_len):
            weights= self.weights[everyNeuron]
            assert len(in_data) == len(weights)
            weighted_input = weights*in_data
            weighted_sum = weighted_input.sum()
            #results.append(PerceptronsLayer.unit_step_function(weighted_sum))
            results.append(weighted_sum)
        return results

    def adjust(self, in_data, calculated_result, target_result):
        """
        accepts a pair of input data and for each neuron a desired output , with the calculated ones
        :param in_data:             the input data
        :param calculated_result:   a list with each neurons output data
        :param target_result:       a list with each neurons target data
        :return:                    the error table for this iteration
        """
#        in_data = PerceptronsLayer.normalise_input_per_input(in_data)#add bias
        errortbl = target_result - calculated_result
        for everyCalculatedResult in range(len(calculated_result)):
            assert len(calculated_result) == len(target_result)
            assert len(self.weights[everyCalculatedResult])==len(in_data)
            assert len(self.weights)==self.neurons_len
            correction = in_data*errortbl[everyCalculatedResult]*self.learn_rate
            self.weights[everyCalculatedResult]=self.weights[everyCalculatedResult]+correction
        return errortbl

p1 = PerceptronsLayer(input_len=2,neurons_len=2,learn_rate=0.1)
p2 = PerceptronsLayer(input_len=2,neurons_len=2,learn_rate=0.1)
p3 = PerceptronsLayer(input_len=2,neurons_len=2,learn_rate=0.1)
network = [p1,p2,p3]
wrapper=NeuralNetwork(network)
print(wrapper(np.array([2,2]),1))
