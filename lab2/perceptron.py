import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

df = df.mask(df.eq('Iris-virginica')).dropna()
del df[1]
del df[3]

plot_dict = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue'}
fig, ax = plt.subplots()
ax.scatter(df[0], df[2], c=df[4].apply(lambda x: plot_dict[x]))

plt.show()
plt.show()


class Adaline():
    def __init__(self):
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        self.gradient = [0.0, 0.0, 0.0]

    def decision_function(self, x):
        return x

    def predict(self, x):
        first = x[0]
        first = first * float(self.weights[0])
        second = x[1]
        second = second * float(self.weights[1])
        return self.decision_function(first + second + self.bias)

    def train(self, x, y, learning_rate=0.0004, iterations=100):
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        self.gradient = [0.0, 0.0, 0.0]
        for _ in range(0, iterations):
            for i in range(0, len(x)):
                predicted_value = self.predict(x[i])
                error_in_prediction = (y[i][0] - predicted_value)
                for j in range(0, len(self.weights)):
                    self.gradient[j + 1] += round(error_in_prediction*x[i][j], 2)
                self.gradient[0] += round(error_in_prediction, 2)

            self.bias += (round(self.gradient[0],2)*learning_rate)
            self.weights[0] += (round(self.gradient[1],2)*learning_rate)
            self.weights[1] += (round(self.gradient[2],2)*learning_rate)

            self.gradient[0] = 0.0
            self.gradient[1] = 0.0
            self.gradient[2] = 0.0
        return [self.bias] + self.weights

x = np.array([df[0],df[2]]).T
y=np.array([df[4]]).T
y = np.where(y == "Iris-setosa" ,-1,1)
random.seed(None)
perceptron = Adaline()
values_for_chart = perceptron.train(x,y)
print(values_for_chart)

plot_dict = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue'}
fig, ax = plt.subplots()
axes = plt.gca()
axes.set_xlim([4,7.2])
values_for_chart = values_for_chart[1]/(-1*values_for_chart[2])*x+values_for_chart[0]/(values_for_chart[2]*-1)
plt.plot(x, values_for_chart, c="black")
ax.scatter(df[0], df[2], c=df[4].apply(lambda x: plot_dict[x]))

plt.show()
plt.show()
