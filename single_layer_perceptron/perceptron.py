import numpy as np

data = np.array([
    [1.00, 0.08, 0.72, 1.0],
    [1.00, 0.10, 1.00, 1.0],
    [1.00, 0.26, 0.58, 1.0],
    [1.00, 0.35, 0.95, 0.0],
    [1.00, 0.45, 0.15, 1.0],
    [1.00, 0.60, 0.30, 1.0],
    [1.00, 0.70, 0.65, 0.0],
    [1.00, 0.92, 0.45, 0.0]
])

test_data = np.array([
    [1.00, 0.42, 0.85, 0.0],
    [1.00, 0.65, 0.55, 0.0],
    [1.00, 0.20, 0.30, 1.0],
    [1.00, 0.20, 1.00, 0.0],
    [1.00, 0.85, 0.10, 1.0]
])

epoch = 50
learning_rate = 0.1

class Perceptron:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.weights = np.zeros(len(train_data[0]) - 1)
        self.bias = 0
    
    def train(self):
        # create generator for shuffling
        rng = np.random.default_rng()

        # run epoch
        for i in range(epoch):

            # run perceptron for each data point
            for idx in range(len(self.train_data)):
                weighted_sum = sum(self.train_data[idx][:-1] * self.weights) + self.bias

                # apply step function to get prediction
                if weighted_sum >= 0:
                    prediction = 1.0
                else:
                    prediction = 0.0
                
                # compare prediction and actual value
                if prediction == self.train_data[idx][-1]:
                    pass
                else:
                    # update weights and bias

                    weights_array = []
                    for w_idx in range(len(self.weights)):

                        weight = self.weights[w_idx] + (learning_rate * (self.train_data[idx][-1] - prediction) * self.train_data[idx][w_idx])

                        weights_array.append(weight)
                    
                    # update weights
                    self.weights = np.array(weights_array)

                    # update bias
                    self.bias = self.bias + (learning_rate * (self.train_data[idx][-1] - prediction))
                
            # shuffle data points in dataset
            rng.shuffle(self.train_data)
        
                
    
p1 = Perceptron(data,test_data)
p1.train()
print(p1.weights)
print(p1.bias)
            