
import numpy as np
import math

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
        in_data=NeuralNetwork.normalisation(in_data)
        result.append(in_data.tolist())

        for i,everyLayer in enumerate(self.layers):
            if(i==0):
                result.append(self.getNthLayerOutput(in_data,i))
            else:
                result.append(self.getNthLayerOutput(NeuralNetwork.normalisation(result[-1]), i))
        return np.array(result[1:])

    @staticmethod
    def normalisation(in_data):
        return np.append(in_data, np.array([1]))  # plus bias input , always at 1

    prev_weights=None
    def adjust(self,in_data,target_result,calculated_result):
        for everyLayer in range(len(self.layers)-1,-1,-1): #traverse the list on reverse
            if(everyLayer==len(self.layers)-1):#we are in the last layer
                prev_weights=(self.layers[everyLayer].terminalAdjust(NeuralNetwork.normalisation(calculated_result[everyLayer - 1]), target_result,
                                                               calculated_result[everyLayer]))

            elif(everyLayer==0):#we are in the first layer , and the input is the networks input
                a=10
                prev_weights=(self.layers[everyLayer].middleAdjust(NeuralNetwork.normalisation(in_data), self.layers[everyLayer + 1],prev_weights,calculated_result[everyLayer]))

                continue

            else:#middle train
                prev_weights=(self.layers[everyLayer].middleAdjust(NeuralNetwork.normalisation(calculated_result[everyLayer-1]),self.layers[everyLayer+1],prev_weights,calculated_result[everyLayer]))

                continue

        return 1









class PerceptronsMultiLayer:

    def __init__(self, input_len,neurons_len, learn_rate,momentum,index:int,weights=None):
        """
        The contructor
        intializes the nessary fields
        :param input_len:  the number of input nodes
        :param learn_rate: the reduce multiplier to eliminate <<jumping>> , optimal at 0.1
        :param weights:    the initial weights , if not given , a 0.5 weight is given for every input node
        """
        self.index=index
        self.learn_rate = learn_rate
        self.momentum=momentum
        self.neurons_len=neurons_len
        self.input_len = input_len
        self.prev_correction=[0]*self.neurons_len
        #self.weights = np.random.randint(0, 10, (neurons_len, input_len)) / 10
        if(weights is None):
            self.weights = np.random.randint(1, 2, (neurons_len, input_len+1)).astype(np.float)
        else:
            self.weights=weights
        self.errortbl=None

    def roundWeights(self):
        for i in range(self.neurons_len):
            self.weights[i]=[np.round(x,4) for x in self.weights[i]]



    @staticmethod
    def sig_activation(x):
        #return x
        #
        return 1/(1+math.pow(math.e,-x))

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
        for everyNeuron in range(self.neurons_len):
            weights= self.weights[everyNeuron]
            assert len(in_data) == len(weights)
            weighted_input = weights*in_data
            weighted_sum = weighted_input.sum()
            results.append(PerceptronsMultiLayer.sig_activation(weighted_sum))
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
        assert len(previous_layer_out)==self.input_len+1


        retval=np.array(self.weights)
        previous_layer_out=np.array(previous_layer_out)
        target_result=np.array(target_result)
        calculated_result_table = np.array(calculated_result_table)
        self.errortbl= (target_result-calculated_result_table)*(1-calculated_result_table)*calculated_result_table
        for everyNode in range(self.neurons_len):
            correction=previous_layer_out* self.errortbl[everyNode]*self.learn_rate + self.prev_correction[everyNode]*self.momentum
            self.prev_correction[everyNode]=correction
            self.weights[everyNode]+=correction

        self.roundWeights()
        return retval

    def middleAdjust(self,previous_layer_out,nextLayer,prev_weights,calculated_result_table):
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
        retval=np.array(self.weights)
        calculated_result_table=np.array(calculated_result_table)
        previous_layer_out=np.array(previous_layer_out)
        for eachNeuron in range(self.neurons_len):
            temp_delta=[]
            """
            find Delta using backpropagation
            """
            for eachNextLayersNeuron in range(nextLayer.neurons_len):
                temp_delta.append(nextLayer.errortbl[eachNextLayersNeuron]*prev_weights[eachNextLayersNeuron][eachNeuron])
            temp_delta=calculated_result_table[eachNeuron]*(1-calculated_result_table[eachNeuron])*np.array(temp_delta).sum()
            correction = previous_layer_out* temp_delta*self.learn_rate + self.prev_correction[eachNeuron]*self.momentum
            self.prev_correction[eachNeuron]=correction
            self.errortbl.append(temp_delta)
            self.weights[eachNeuron]+=correction

        self.roundWeights()
        return retval




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
p1 = PerceptronsMultiLayer(input_len=2, neurons_len=2, learn_rate=0.5,momentum=0.6,index=0,weights=np.array([[-0.1558,0.2829,0.8625],[-0.5060,-0.8644,0.8350]]).astype(np.float)) #remember , add bias manually
p2 = PerceptronsMultiLayer(input_len=2, neurons_len=1, learn_rate=0.5,momentum=0.6,index=1,weights=np.array([[-0.4304,0.4812,0.0365]]).astype(np.float))
network = [p1,p2]
wrapper=NeuralNetwork(network)

epochs=1000

data=np.array([[0,0],[0,1],[1,0],[1,1]])
target=np.array([[0],[1],[1],[0]])




#r=wrapper(data[0])
#wrapper.adjust(data[0],target[0],r)

#r=wrapper(data[1])
#wrapper.adjust(data[1],target[1],r)

#r=wrapper(data[2])
#wrapper.adjust(data[2],target[2],r)

print(p1.weights)
print("\n")
print(p2.weights)

#"""
def step(x):
    return x
    if(x>0.5):return 1
    return 0


for eachDataset in range(len(data)):
    r = wrapper(data[eachDataset])
    print(step(r[-1][0]))

print("----------")
for i in range(epochs):
    for eachDataset in range(len(data)):
        r = wrapper(data[eachDataset])
        wrapper.adjust(data[eachDataset], target[eachDataset], r)

for eachDataset in range(len(data)):
    r = wrapper(data[eachDataset])
    print(step(r[-1][0]))
#"""
