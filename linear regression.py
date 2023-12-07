import numpy as np
import matplotlib.pyplot as plt
def J_function(a1, a0, x, y):
    Error = 0
    for i in range(0, len(x)):
        Error += (y[i]-(a1*x[i]+a0))**2
    return Error/float(len(x))
def  gd(a0, a1, x, y, learning_rate, num_iterations):
    N = float(len(x))
    for j in range(num_iterations): # repeat for num_iterations
        a0_gradient = 0
        a1_gradient = 0
        for i in range(0, len(x)):
            a0_gradient += -(2/N) * (y[i] - ((a1 * x[i]) + a0))
            a1_gradient += -(2/N) * x[i] * (y[i] - ((a1 * x[i]) + a0))
        a0 -= (learning_rate * a0_gradient)
        a1 -= (learning_rate * a1_gradient)
        print('Actual Error:', J_function(a1, a0, x, y))
    return [a0, a1]
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y =  3 * x +4+ np.random.randn(100, 1)
plt.scatter(x, y)
learning_rate = 0.01
initial_a0 = 0
initial_a1 = 0
num_iterations= 1000
print('Starting error:', J_function(initial_a1, initial_a0, x, y))
[a0, a1] =  gd(initial_a0, initial_a1, x, y, learning_rate, num_iterations)
print('alpha0:', a0)
print('alpha1:', a1)
print('Final Error:', J_function(a1, a0, x, y))
results = [(a1 * x[i]) + a0 for i in range(len(x))]
plt.scatter(x, y,c='b')
plt.plot(x, results, color='r')
