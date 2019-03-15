# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:25:06 2019

@author: Pranjall
"""

"""                         Building an ANN that predicts the users personality type based on his responses to questions                """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(x, 0, x)

def sigmoid_derivative(P):
    return P * (1 - P)

def ReLU_derivative(P):
    P[P <= 0] = 0
    P[P > 0] = 1
    return P


class NeuralNetwork:
    
    def __init__(self, x, y):
        
        self.input = x
        self.y = y
        self.nodes_in_first_layer = 5
        self.nodes_in_second_layer = 5
        self.nodes_in_output_layer = 6
        self.output = np.zeros(y.shape)
        self.error_values = []
        
        upper_limit = 0.5
        lower_limit = -0.5
        
        self.weights1 = np.random.uniform(upper_limit, lower_limit, (self.input.shape[1], self.nodes_in_first_layer))
        self.weights2 = np.random.uniform(upper_limit, lower_limit, (self.nodes_in_first_layer, self.nodes_in_second_layer))
        self.weights3 = np.random.uniform(upper_limit, lower_limit, (self.nodes_in_second_layer, self.nodes_in_output_layer))
        
        self.bias1 = np.random.uniform(upper_limit, lower_limit, (1, 1))
        self.bias2 = np.random.uniform(upper_limit, lower_limit, (1, 1))
        self.bias3 = np.random.uniform(upper_limit, lower_limit, (1, 1))
        
    def forwardprop(self):
        
        self.layer1 = ReLU(np.dot(self.input, self.weights1) + self.bias1)
        self.layer2 = ReLU(np.dot(self.layer1, self.weights2) + self.bias2)
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3) + self.bias3)
        
        return self.layer3
    
    def error(self):
        return -(1 / self.output.shape[0]) * np.sum((self.y * np.log(self.output)) + ((1 - self.y) * np.log(1 - self.output)))
     
    def backprop(self):
        
        learning_rate = 0.1
        
        d_Propogation3 = self.output - self.y
        d_weights3 = (1 / self.output.shape[0]) * (np.dot(d_Propogation3.T, self.layer2))
        d_bias3 = (1 / self.output.shape[0]) * (np.sum(d_Propogation3))
        
        d_Propogation2 = np.dot(d_Propogation3, self.weights3.T) * ReLU_derivative(self.layer2)
        d_weights2 = (1 / self.output.shape[0]) * (np.dot(d_Propogation2.T, self.layer1))
        d_bias2 = (1 / self.output.shape[0]) * (np.sum(d_Propogation2))
        
        d_Propogation1 = np.dot(d_Propogation2, self.weights2.T) * ReLU_derivative(self.layer1)
        d_weights1 = (1 / self.output.shape[0]) * (np.dot(d_Propogation1.T, self.input))
        d_bias1 = (1 / self.output.shape[0]) * (np.sum(d_Propogation1))
        
        self.weights3 = self.weights3 - learning_rate * d_weights3.T
        self.bias3 = self.bias3 - learning_rate * d_bias3
        
        self.weights2 = self.weights2 - learning_rate * d_weights2.T
        self.bias2 = self.bias2 - learning_rate * d_bias2
        
        self.weights1 = self.weights1 - learning_rate * d_weights1.T
        self.bias1 = self.bias1 - learning_rate * d_bias1
        
    def simulate(self):
        self.output = self.forwardprop()
        self.error_values.append(self.error())
        self.backprop()
        
        return self.output, self.error_values
    
    def run(self, test):
        
        self.layer1 = ReLU(np.dot(test, self.weights1) + self.bias1)
        self.layer2 = ReLU(np.dot(self.layer1, self.weights2) + self.bias2)
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3) + self.bias3)
        
        return self.layer3, self.weights1
    



if __name__ == '__main__':
    
    #importing data.
    dataset = pd.read_csv("Amazon_user_data.csv")
    X = dataset.iloc[:, 1 : 6]
    Y = dataset.iloc[:, 6]
    
    #normalizing data.
    from sklearn.preprocessing import StandardScaler
    X_sc = StandardScaler()
    X = X.astype('float')
    X = pd.DataFrame(X_sc.fit_transform(X))
    
    #categorical data.
    Y = pd.get_dummies(Y, columns=['PersonalityType'], prefix = ['Type']) 
    
    #converting to numpy arrays.
    X = np.array(X)
    Y = np.array(Y)
    m = len(Y)
    
    #feature scaling.
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = X.astype('float')
    X = sc.fit_transform(X)
    
    #splitting data.
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    
    #simulating the Neural Network.
    NN = NeuralNetwork(X_train, Y_train)
    iterations = 800
    
    #monitoring execution time.
    start_time = time.time()

    for i in range(0, iterations):
        print('Iteration: ' + str(i+1) + ' / ' + str(iterations))
        output, error_values = NN.simulate()
        
    #plotting graph.
    interation_values = np.arange(1, iterations + 1)
    plt.plot(interation_values, error_values, c = 'red')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Convergence of error')
    plt.show()
        
    #storing execution time.
    execution_time = time.time() - start_time
    
    #simulating the Neural Network on test data.
    output, W = NN.run(X_test)
    result = np.zeros_like(output)
    result[np.arange(len(output)), output.argmax(1)] = 1
    
    #calculating accuracy.
    count = 0
    for i in range(len(result)):
        if np.array_equal(result[i], Y_test[i]):
            count += 1
    accuracy = count / len(result)
    print()
    print('Prediction Accuracy: ' + str(accuracy))
    print('Execution Time: ' + str(execution_time))
