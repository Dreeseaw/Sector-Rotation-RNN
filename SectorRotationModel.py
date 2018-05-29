'''
Sector Rotation Model
William Dreese 2018

Model: IndRNN, based off of 
'Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN'
(https://arxiv.org/abs/1803.04831)
Hidden Layers: 2
Hidden Layer Sizes: 10
Act: tanh
'''

import numpy as np
import math
from SectorRotationDataPrep import DataPrep 
np.random.seed(0)

class Model:

    def __init__(self,train):

        self._data = train

        #hyper-parameters
        self._alpha = 2e-2
        self._hidden1Size = 20
        self._hidden2Size = 20
        self._epochs = 10000
        self._batch_size = 20
        
        #parameters
        self._INtoHID1 = 2*np.random.random((self._hidden1Size,len(self._data[0]))) - 1
        self._HID1toHID2 = 2*np.random.random((self._hidden2Size,self._hidden1Size)) - 1
        self._HID2toOUT = 2*np.random.random((len(self._data[0]),self._hidden2Size)) - 1
        self._HIDDEN1 = (2*np.random.random(self._hidden1Size) - 1)*(1/math.sqrt(2.0))
        self._HIDDEN2 = 2*np.random.random(self._hidden2Size) - 1

    def trainModel(self):
        #make batchs from our training data
        newData = list()
        for tt in range(len(self._data)-1):
            newData.append([self._data[tt],self._data[tt+1]])
        batchs = list()
        for t in range(len(newData)-self._batch_size):
            batchs.append(newData[t:t+self._batch_size])

        #train model
        for a in range(self._epochs):
            for b in batchs:
                e = self.trainingPass(b)
            if a % 10 == 0:
                print ("Epoch: ",str(a),", Error: ",str(e))

    def trainingPass(self,batch):
        
        #values stored for backprop through time
        error = list()
        acts1 = list()
        acts2 = list()
        outs = list()
        #1st time step inits
        acts1.append(np.zeros(self._hidden1Size))
        acts2.append(np.zeros(self._hidden2Size))
        totalError = 0
        
        for a in range(len(batch)):
            #layer 1 pass
            layer1 = np.dot(batch[a][0],self._INtoHID1.T)
            layer1 += acts1[-1]*self._HIDDEN1
            layer1 = np.tanh(layer1)
            acts1.append(layer1)
            #layer 2 pass
            layer2 = np.dot(layer1,self._HID1toHID2)
            layer2 += acts2[-1]*self._HIDDEN2
            layer2 = np.tanh(layer2)
            acts2.append(layer2)
            #output layer pass
            outLayer = np.dot(layer2,self._HID2toOUT)
            outLayer = np.tanh(outLayer)
            outs.append(outLayer)
            #calc MSE for each output node
            errors = batch[a][1] - outLayer
            errors = pow(errors,2)
            errors /= 2
            error.append(errors)
            totalError += sum(errors)
            
        inUpdate = np.zeros_like(self._INtoHID1)
        midHidUpdate = np.zeros_like(self._HID1toHID2)
        outUpdate = np.zeros_like(self._HID2toOUT)
        hidden1Update = np.zeros_like(self._HIDDEN1)
        hidden2Update = np.zeros_like(self._HIDDEN2)

        hidden1futureDelta = np.zeros_like(self._HIDDEN1)
        hidden2futureDelta = np.zeros_like(self._HIDDEN2)

        #need to add gradient clipping
        for a in range(len(batch)):
            #output layer gradients
            dy = outs[1-a]-batch[1-a][1] #(dE/dOut)
            dOn = 1 - outs[1-a]*outs[1-a] #(dOut/dNet)
            out_delta = dy*dOn
            out_grads = [out_delta*acts2[1-a][x] for x in range(self._hidden2Size)]
            outUpdate += out_grads
            #hidden layer 2 gradients
            h2_delta = sum(out_grads) + hidden2futureDelta
            h2_delta = h2_delta*(1-(acts2[1-a] * acts2[1-a]))
            midHid_grads = [h2_delta*acts1[1-a][x] for x in range(self._hidden1Size)]
            midHidUpdate += midHid_grads
            hidden2Update += h2_delta*acts2[2-a]
            hidden2futureDelta = h2_delta
            #hidden layer 1 gradients
            ins = np.array(batch[1-a][0])
            h1_delta = sum(midHid_grads) + hidden1futureDelta
            h1_delta = h1_delta*(1-(ins*ins))
            in_grads = [h1_delta*batch[1-a][0][x] for x in range(len(batch[0][0]))]
            inUpdate += in_grads
            hidden1Update += h1_delta*acts1[2-a]
            hidden1futureDelta = h1_delta

        self._INtoHID1 -= inUpdate * self._alpha
        self._HID1toHID2 -= midHidUpdate * self._alpha
        self._HID2toOUT -= outUpdate * self._alpha
        self._HIDDEN1 -= hidden1Update * self._alpha
        self._HIDDEN2 -= hidden2Update * self._alpha

        inUpdate *= 0
        midHidUpdate *= 0 
        outUpdate *= 0
        hidden1Update *= 0
        hidden2Update *= 0

        return totalError

dp = DataPrep()
mod = Model(dp.newDelta)
mod.trainModel()


