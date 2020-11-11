import numpy as np
import matplotlib.pyplot as plt
from activation_functions import LogisticFunction, LogisticDerivativeFunction

class MLP:
    
    def __init__(self, input_values, output_values, layers, activation_function=LogisticFunction, derivative_function=LogisticDerivativeFunction, learning_rate=0.1, precision=1e-6):
       ones_column = np.ones((len(input_values), 1)) * -1
       self.input_values = np.append(ones_column, input_values, axis=1)
       self.output_values = output_values
       self.learning_rate = learning_rate
       self.precision = precision
       self.activation_function = activation_function
       self.derivative_function = derivative_function
       self.EqmPlot = []
       
       self.W = []
       neuron_input = self.input_values.shape[1]
       for i in range(len(layers)):
           self.W.append(np.random.rand(layers[i], neuron_input))
           neuron_input = layers[i] + 1
          
       self.epochs = 0
       
    def train(self):
        
        error = True
        
        while error:
            print(f'[EPOCH] {self.epochs}')
            error = False
            
            eqm_previous = self.eqm()
            
            for x, d in zip(self.input_values, self.output_values):
            
                I1 = np.dot(self.W[0], x)
                Y1 = np.zeros(I1.shape)
                for i in range(Y1.shape[0]):
                    Y1[i] = self.activation_function.g(I1[i])
                Y1 = np.append(-1, Y1)
                
                I2 = np.dot(self.W[1], Y1)
                Y2 = np.zeros(I2.shape)
                for i in range(Y2.shape[0]):
                    Y2[i] = self.activation_function.g(I2[i])
                    
                Grad2 = (d - Y2) * self.derivative_function.dg(I2)
                self.W[1] = self.W[1] + (self.learning_rate * Grad2[:,np.newaxis] * Y1)

                Soma = sum(Grad2[:,np.newaxis] * self.W[1]) 
                Grad1 = Soma[1:] * self.derivative_function.dg(I1)
                self.W[0] = self.W[0] + (self.learning_rate * Grad1[:,np.newaxis] * x)
                
            eqm_actual = self.eqm()
            self.epochs += 1
            EqmE = abs(eqm_actual - eqm_previous)
            self.Plot(eqm_previous, EqmE)
            if EqmE > self.precision:
                error = True
                            
        
    def eqm(self):
        
        eq = 0
        
        for x, d in zip(self.input_values, self.output_values):
            I1 = np.dot(self.W[0], x)
            Y1 = np.zeros(I1.shape)
            for i in range(Y1.shape[0]):
                Y1[i] = self.activation_function.g(I1[i])
            Y1 = np.append(-1, Y1)
            
            I2 = np.dot(self.W[1], Y1)
            Y2 = np.zeros(I2.shape)
            for i in range(Y2.shape[0]):
                Y2[i] = self.activation_function.g(I2[i])
                
            eq += 0.5 * sum((d - Y2) ** 2)
            
        return eq/len(self.output_values)
        
    def evaluate(self, x):
        x = np.append(-1, x)
        I1 = np.dot(self.W[0], x)
        Y1 = np.zeros(I1.shape)
        for i in range(Y1.shape[0]):
            Y1[i] = self.activation_function.g(I1[i])
        Y1 = np.append(-1, Y1)
                
        I2 = np.dot(self.W[1], Y1)
        Y2 = np.zeros(I2.shape)
        for i in range(Y2.shape[0]):
            Y2[i] = self.activation_function.g(I2[i])
            
        for i in range(Y2.shape[0]):
            if(Y2[i] >= 0.5):
                Y2[i] = 1
            elif(Y2[i] < 0.5):
                Y2[i] = 0
            
        return Y2
        
    def Plot(self, eqm1, eqme):    
        self.EqmPlot.append(eqm1)   
        
        if(eqme <= self.precision):
            x = np.arange(self.epochs)
            plt.plot(x, self.EqmPlot)        
            plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        