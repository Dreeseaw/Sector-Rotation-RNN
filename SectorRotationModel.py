'''
Sector Rotation Model
William Dreese 2018

Current Version (1.0.0)
Model: RNN
Hidden Layer Size: 10
'''
import copy, numpy as np
from SectorRotationDataPrep import DataPrep 
np.random.seed(0)

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

class Model:

    def __init__(self,train,test):
        self._train = train
        self._test = test
        self._alpha = 0.2
        self._hiddenSize = 10
        self._epochs = 1000
              
        self._INtoHID = 2*np.random.random((len(self._train[0][0]),self._hiddenSize)) - 1
        self._HIDtoOUT = 2*np.random.random((self._hiddenSize,len(self._train[0][0]))) - 1
        self._HIDDEN = 2*np.random.random((self._hiddenSize,self._hiddenSize)) - 1

    def trainModel(self):
        for a in range(self._epochs):
            #forward pass for all training data (67), calc loss
            acts,loss,overallLoss = self.feedforward()
            #backprop on all weights
            self.updateWeights(acts,loss)

    def testModel(self):
        acts = list()
        acts.append(np.zeros(self._hiddenSize))
        for a in range(len(self._test)):
            layer1 = np.dot(self._test[a][0],self._INtoHID)
            layer1 += np.dot(acts[-1],self._HIDDEN)
            layer1 = sigmoid(layer1)
            acts.append(layer1)
            outLayer = np.dot(layer1,self._HIDtoOUT)
            outLayer = sigmoid(outLayer)
            print (outLayer,self._test[a][1])
            

    def updateWeights(self,acts,loss):
        
        inUpdate = np.zeros_like(self._INtoHID)
        outUpdate = np.zeros_like(self._HIDtoOUT)
        hiddenUpdate = np.zeros_like(self._HIDDEN)
        
        future_layer_1_delta = np.zeros(self._hiddenSize)

        for a in range(len(self._train)):

            X = np.array(self._train[a][0])
            layer_1 = acts[a-1]
            prev_layer_1 = acts[a-2]
            layer_2_delta = loss[a-1]
            layer_1_delta = (future_layer_1_delta.dot(self._HIDDEN.T) + layer_2_delta.dot(self._HIDtoOUT.T)) * sigmoid_output_to_derivative(layer_1)
            
            outUpdate += layer_1.T.dot(layer_2_delta)
            hiddenUpdate += prev_layer_1.T.dot(layer_1_delta)
            inUpdate += X.T.dot(layer_1_delta)
            future_layer_1_delta = layer_1_delta

        self._INtoHID += inUpdate * self._alpha
        self._HIDtoOUT += outUpdate * self._alpha
        self._HIDDEN += hiddenUpdate * self._alpha
        
        inUpdate *= 0
        outUpdate *= 0
        hiddenUpdate *= 0

    def feedforward(self):
        overallLoss = 0
        loss = list()
        acts = list()
        acts.append(np.zeros(self._hiddenSize))
        for a in range(len(self._train)):
            #1st layer pass
            layer1 = np.dot(self._train[a][0],self._INtoHID)
            layer1 += np.dot(acts[-1],self._HIDDEN)
            layer1 = sigmoid(layer1)
            acts.append(layer1)
            #2nd layer pass
            outLayer = np.dot(layer1,self._HIDtoOUT)
            outLayer = sigmoid(outLayer)
            #calc total error
            layer_2_error = self._train[a][1] - outLayer
            loss.append((layer_2_error)*sigmoid_output_to_derivative(outLayer))
            overallLoss += np.abs(layer_2_error[0])
        return acts,loss,overallLoss

dp = DataPrep()
mod = Model(dp._trainingData,dp._testData)
mod.trainModel()
mod.testModel()



