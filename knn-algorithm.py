
import math
import numpy as np
import matplotlib.pyplot as plt
def knn(data, test, k, distance_fn):
    D_set = []
    for index, example in enumerate(data):
        distance = distance_fn(example, test)
        D_set.append((distance, index))
        sorted_D_set = sorted(D_set)
    F_set = sorted_D_set[:k]
 
    return F_set 
def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(2):
         sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)
my_data = [
       [18, 10],
       [19, 15],
       [21, 15],
       [22, 10],
       [23, 12],
       [25, 9],
       [27, 17],
       [29, 16],
       [31, 11],
       [45, 20],
       [46, 21],
       [47, 24],
       [48, 19],
       [50, 25],
       [51, 18],
       [52, 20],
       [53, 19],
    ]


Test_X,Test_Y = input("Enter your test data: ").split()
Test =[int(Test_X),int(Test_Y)]


k_nearest_neighbors = knn(
        my_data, Test, k=3, distance_fn=euclidean_distance
    )
res=np.array(k_nearest_neighbors)[:,1].astype(int)
res=res.tolist()
print("the k neighbors are:")
for i in res:
    print(my_data[i])
together=my_data
together.append(Test)
mat=np.array(together)
l1=['r','r','r','r','r','r','r','r','r','b','b','b','b','b','b','b','b','0']
plt.scatter(mat[:,0],mat[:,1],marker='+',color=l1 )
plt.show()