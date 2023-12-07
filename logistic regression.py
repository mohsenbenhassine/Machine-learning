import numpy as np
import matplotlib.pyplot as plt

def sigmoid(yi):
    return 1 / (1 + np.exp(-yi))
def log_likelihood(features, target, weights):
    yi = np.dot(features, weights)
    ll = np.sum( target*yi - np.log(1 + np.exp(yi)) )
    return ll
def logistic_regression(features, target, num_iterations, learning_rate):
    
    bias = np.ones((features.shape[0], 1))
    features = np.hstack((bias, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_iterations):
        yi = np.dot(features, weights)
        predictions = sigmoid(yi)
        output_error = target - predictions
        gradient = np.dot(features.T, output_error)
        weights += learning_rate * gradient
        print ('Actual ll=',log_likelihood(features, target, weights))
        
    return weights
np.random.seed(1)
num_observations =500

class1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
class2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

mix_cl1_cl2 = np.vstack((class1, class2)).astype(np.float32)
c1=np.zeros(num_observations)
c2=np.ones(num_observations)
color1=np.full( num_observations,'g')
color2=np.full( num_observations,'r')

targets = np.hstack((c1,c2))
colors2 = np.hstack((color1,color2))
xaxis=np.sort(mix_cl1_cl2[:, 0])
yaxis=mix_cl1_cl2[:, 1]
plt.figure(figsize=(12,8))
plt.scatter(mix_cl1_cl2[:, 0], yaxis,
             c = colors2, alpha = .4)



weights = logistic_regression(mix_cl1_cl2, targets,
                     num_iterations = 30000, learning_rate = 5e-5)
a0=weights[0]
a1=weights[1]
a2=weights[2]
print('a0=',a0)
print('a1=',a1)
print('a2=',a2)

yaxis=-(a0+a1*xaxis)/a2
plt.plot(xaxis,yaxis,linewidth=4.0)
plt.show()