import time  
import numpy as np
import matplotlib.pyplot as plt
# define the activation sigmoid function
def Sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
# define the derivative of sigmoid function  
def SigmoidDeriv(z):
    return z * (1 - z)
# define the ANN class
class NeuralNetwork:
    def __init__(self, x, y):
        # X (3*1) vector 
        self.input = x
        # Between Layers weights W1(3*4) and W2(4*1) 
        self.W1 = np.random.rand(self.input.shape[1], 4)
        self.W2 = np.random.rand(4, 1)
        # Real output 
        self.y = y
        # Estimated probaility of y, >0.5 y=1 else y=0
        self.output = np.zeros(self.y.shape)
        # initialize the loss function
        self.Loss=[]
# feed forward function
    def feedforward(self):
        self.layer1 = Sigmoid(np.dot(self.input, self.W1))
        self.output = Sigmoid(np.dot(self.layer1, self.W2))
# BP and gradient computation
    def backpropagation(self):
        # neurons errors for output layer
        e3=2*(self.y - self.output) * SigmoidDeriv(self.output)
        # W2 errors
        d_W2 = np.dot(self.layer1.T,e3)
        # neurons errors for hidden layer 1
        e2=np.dot( e3, self.W2.T) * SigmoidDeriv(self.layer1)
        # W1 errors
        d_W1 = np.dot(self.input.T, e2)
        # update weights W1 and W2
        self.W1 += d_W1
        self.W2 += d_W2
        # compute the actual sum of errors throughout the network
        error=  sum(self.y-self.output)**2
        self.Loss.append(error)
# predict output for new data
def predict(vect,W1,W2):
         L1 = Sigmoid(np.dot(vect,W1))
         Out =Sigmoid(np.dot(L1,W2))
         return  Out
#--------------------Main program ---------------------------
# Initialize input data X
X = np.array([[0, 0,0],
                  [0,0, 1],
                  [0,1,0],
                  [0, 1, 1],
                  [1, 0, 0],[1, 0, 1],[1, 1, 0]
                  ])

# outputs: Y= x1 xor x2 xor x3  
Y = np.array([[0], [1], [1], [0], [1],[0],[0]])
np.random.seed(1)
nn = NeuralNetwork(X, Y)

print(" The program is training...............................................\n ")
time.sleep(2)
for i in range(10000):
        nn.feedforward()
        nn.backpropagation()
print('----------Final outputs: \n\n'+str(nn.output*Y)+'\n')
print('----------input weights:\n\n'+str(nn.W1)+'\n')
print('----------Layer 1 weights:\n\n'+str(nn.W2))
time.sleep(1)
print('The program is predicting now......... ')
Inp_Pre=np.array([1, 1,1])
print('---------- Input value :',Inp_Pre)
time.sleep(1)

Res_Pre=predict(Inp_Pre,nn.W1,nn.W2)
print('---------- Predicted Result is:',Res_Pre  )
# print Loss values for the the first and final epochs
print('Loss values:\n','initial value',nn.Loss[0],'\n','Final value:',nn.Loss[len(nn.Loss)-1])
plt.plot(nn.Loss)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")