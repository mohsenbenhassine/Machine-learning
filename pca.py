#There are 150 observations with 4 features each (sepal length, #sepal width, petal length, petal width).
#There are 50 observations of each species (setosa, versicolor, #virginica).


import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('c:/users/hp/iris.csv', header=None, sep=',')
# X is the features vector
X = df.iloc[:, :-1]
# labs is the labels vector
Labs = df.iloc[:,-1]
Labs = pd.DataFrame(df.iloc[:,-1])
Labs.columns = ['label']
#step 1 normalize the data set
X = X.sub(X.mean(axis=0), axis=1)
#step 2 calculate  feature matrix covariance
mat = np.asmatrix(X)
C = np.cov( mat.T)
# step 3 Finding eigen values and eigen vectors for C
eigVals, eigVec = np.linalg.eig(C)
# step 4 Sort the eigen values and eigen vectors
sorted_index = eigVals.argsort()[::-1] 
eigVals = eigVals[sorted_index]
eigVec = eigVec[:,sorted_index]
# choose the largest 2 eigen values 
eigVec = eigVec[:,:2]
# step 5 find the new data set in reduced dimensions (2)
Xnew =  mat.dot(eigVec)
#Concatenate transformed data set with labels
new_df = np.hstack((Xnew, Labs))
new_df = pd.DataFrame(new_df)

new_df.columns = ['x','y','label']
#plot data in the new subspace
groups = new_df.groupby('label')
figure, axes = plt.subplots()
axes.margins(0.05)
for nl, group in groups:
    axes.plot(group.x, group.y, marker='o', linestyle='' , label=nl)
    axes.set_title(" Iris Data with 2 features")
axes.legend()
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()