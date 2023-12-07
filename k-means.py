from copy import deepcopy
import numpy as np
 
from matplotlib import pyplot as plt
np.random.seed(0)
# Euclidean Distance Function
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

plt.style.use('ggplot')
mu1  = [0,0]
cov  = [[1, 0], [0, 1]]

s1 = np.random.multivariate_normal(mu1, cov , 1000)
mu2  = [5,5]
s2 = np.random.multivariate_normal(mu2, cov , 1000)
mu3  = [8,0]
s3 = np.random.multivariate_normal(mu3, cov , 1000)
X = np.vstack((s1, s2))
X = np.vstack((X, s3))

f1 = X[:,0]
f2=X[:,1]
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='red')
k = 3
C_x = np.random.randint(0, np.max(X) , size=k)
C_y = np.random.randint(0, np.max(X) , size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
plt.scatter(f1, f2, c='red', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
error = dist(C, C_old, None)
np.set_printoptions(3)
while error != 0:
    print(C)
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
       points = [X[j] for j in range(len(X)) if clusters[j] == i]
       C[i] = np.mean(points, axis=0)
       error = dist(C, C_old, None)

colors = ['r', 'g', 'b']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')

    