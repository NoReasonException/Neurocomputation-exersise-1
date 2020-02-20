
import numpy as np


class NeuralNetwork:
    def __init__(self,layers:iter):
        NeuralNetwork.assertAllTrue([isinstance(x, PerceptronsMultiLayer) for x in layers])
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


    def getNthLayerOutput(self,in_data:iter,n:int):
        """
        receives the output of n-1 layer and calculates the output of nth layer
        :param input: the output of n-1 layer , used as an input
        :param n: the index of the layer we want to calculate
        :return: a list with the outputs on nth layer
        """
        nthLayer=self.layers[n]
        return nthLayer.__call__(in_data)


    def __call__(self, in_data):
        result=[]
        result.append(in_data.tolist())
        for i,everyLayer in enumerate(self.layers):
            if(i==0):
                result.append(self.getNthLayerOutput(in_data,i))
            else:
                result.append(self.getNthLayerOutput(result[-1], i))
        return np.array(result[1:])


    def adjust(self,in_data,target_result,calculated_result):
        self.errorTbl=[]
        for everyLayer in range(len(self.layers)-1,-1,-1): #traverse the list on reverse
            if(everyLayer==len(self.layers)-1):#we are in the last layer
                self.errorTbl.append(self.layers[everyLayer].terminalAdjust(calculated_result[everyLayer - 1], target_result,
                                                               calculated_result[everyLayer]))

            elif(everyLayer==0):#we are in the first layer , and the input is the networks input
                self.errorTbl.append(self.layers[everyLayer].middleAdjust(in_data,self.layers[everyLayer+1]))
                continue

            else:#middle train
                self.errorTbl.append(self.layers[everyLayer].middleAdjust(calculated_result[everyLayer-1],self.layers[everyLayer+1]))
                continue

        return 1









class PerceptronsMultiLayer:

    def __init__(self, input_len,neurons_len, learn_rate,index:int,weights=None):
        """
        The contructor
        intializes the nessary fields
        :param input_len:  the number of input nodes
        :param learn_rate: the reduce multiplier to eliminate <<jumping>> , optimal at 0.1
        :param weights:    the initial weights , if not given , a 0.5 weight is given for every input node
        """
        self.index=index
        self.learn_rate = learn_rate
        self.neurons_len=neurons_len
        self.input_len = input_len
        #self.weights = np.random.randint(0, 10, (neurons_len, input_len)) / 10
        if(weights is None):
            self.weights = np.random.randint(1, 2, (neurons_len, input_len+1)).astype(np.float)
        else:
            self.weights=weights
        self.errortbl=None

    @staticmethod
    def normalise_input_per_input(in_data):
        return np.append(in_data, np.array([1]))  # plus bias input , always at 1


    @staticmethod
    def linear_activation(x):
        """
        This is the activation function , is a plain linear activation
        (defined as static)
        :param x: the weighted sum
        """
        return x

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
        in_data=PerceptronsMultiLayer.normalise_input_per_input(in_data) #add bias
        for everyNeuron in range(self.neurons_len):
            weights= self.weights[everyNeuron]
            print("neuron "+str(everyNeuron)+" with weights "+str(weights))
            assert len(in_data) == len(weights)
            weighted_input = weights*in_data
            weighted_sum = weighted_input.sum()
            results.append(PerceptronsMultiLayer.linear_activation(weighted_sum))
        return results


    def terminalAdjust(self,previous_layer_out,target_result, calculated_result_table):
        """
                                 O-\
                                    0--     Error = Target-Calculated
                                 O-/        Input = previous layers output


        :param previous_layer_out:
        :param target_result:
        :param calculated_result_table:
        :return:
        """
        assert len(target_result)==len(calculated_result_table)
        assert len(target_result)==self.neurons_len
        assert len(previous_layer_out)==self.input_len
        previous_layer_out=PerceptronsMultiLayer.normalise_input_per_input(previous_layer_out) #add bias
        previous_layer_out=np.array(previous_layer_out)         #in order to do scalar multiplication(see below)
        target_result=np.array(target_result)
        calculated_result_table = np.array(calculated_result_table)
        self.errortbl= target_result - calculated_result_table
        for everyNode in range(self.neurons_len):
            correction=previous_layer_out*self.errortbl[everyNode]*self.learn_rate
            self.weights[everyNode]+=correction


        return self.errortbl

    def middleAdjust(self,previous_layer_out,nextLayer):
        """                     #next_layer_errortbl,next_layer_weights
             ---0----0
                 \  / \
                  \/   0
                 / \  /
             ---0---0

        :param previous_layer_out:
        :param next_layer_errortbl:
        :param next_layer_weights:
        :return:
        """
        self.errortbl=[]
        previous_layer_out=PerceptronsMultiLayer.normalise_input_per_input(previous_layer_out) #add bias
        previous_layer_out=np.array(previous_layer_out)

        for eachNeuron in range(self.neurons_len):
            temp_delta=[]
            """
            find Delta using backpropagation
            """
            for eachNextLayersNeuron in range(nextLayer.neurons_len):
                temp_delta.append(nextLayer.errortbl[eachNextLayersNeuron]*nextLayer.weights[eachNextLayersNeuron][eachNeuron])
            temp_delta=np.array(temp_delta).sum()
            correction = previous_layer_out*temp_delta*self.learn_rate
            self.errortbl.append(temp_delta)
            self.weights[eachNeuron]=self.weights[eachNeuron]+correction
        return self.errortbl




    """
    def adjust(self, in_data, calculated_result, target_result):
#        in_data = PerceptronsLayer.normalise_input_per_input(in_data)#add bias
        self.errortbl = target_result - calculated_result
        for everyCalculatedResult in range(len(calculated_result)):
            assert len(calculated_result) == len(target_result)
            assert len(self.weights[everyCalculatedResult])==len(in_data)
            assert len(self.weights)==self.neurons_len
            correction = in_data*self.errortbl[everyCalculatedResult]*self.learn_rate
            self.weights[everyCalculatedResult]=self.weights[everyCalculatedResult]+correction
        return self.errortbl
"""
#remember , x3 is the bias! not x0
p1 = PerceptronsMultiLayer(input_len=2, neurons_len=2, learn_rate=0.5,index=0,weights=np.array([[-0.1558,0.2829,0.8625],[-0.5060,-0.8644,0.8350]]).astype(np.float)) #remember , add bias manually
p2 = PerceptronsMultiLayer(input_len=2, neurons_len=1, learn_rate=0.5,index=1,weights=np.array([[-0.4304,0.4812,0.0365]]).astype(np.float))
network = [p1,p2]
wrapper=NeuralNetwork(network)

epochs=70

data=np.array([[0,0],[0,1],[1,0],[1,1]])
target=np.array([[0],[1],[1],[0]])

print(p1(np.array([0,0])))

"""
for i in range(epochs):
    for eachDataset in range(len(data)):
        r=wrapper(data[eachDataset]) #result
        print(r)
        wrapper.adjust(data[eachDataset],target[eachDataset],r)
    print("------------------------------------------------------")

print("----------")
for eachDataset in range(len(data)):
    r=wrapper(data[eachDataset]) #result
    PerceptronsMultiLayer.linear_activation(r[-1][0])
"""

"""
for i in range(epochs):
    for everyData in range(len(data)):
        result=wrapper(np.array([2,2]))
        wrapper.adjust(data[everyData],target[everyData],result)
        print(result)
"""
