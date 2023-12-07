 
import numpy as np
def freq_num (N,data,fd,xp):
    lle = np.zeros((features_dim,class_num), dtype=np.int) 
    prior = np.zeros(class_num, dtype=np.int)
    for i in range(N):
         y = int(data[i][features_dim])
         prior[y] += 1
         for j in range(features_dim):
           if data[i][j] == X_P[j]:
               lle[j][y] += 1
    return lle,prior

def evid_terms(cn,fd,lle,prior,n):
    post_prob = np.zeros(class_num, dtype=np.float32) 
    for k in range(class_num):
       llx = 1.0
       for j in range(features_dim):
           llx *= lle[j][k] / (prior[k] )
       llx *= prior[k] / N
       post_prob[k] = llx
    return post_prob
data = np.loadtxt("c:/users/hp/forecast.txt",dtype=np.str, delimiter=" ")
print("Forcast Data : ")
print("outlook  temp humidity windy ==> decision")
for i in range(14):
    print(data[i])
print(" \n")

features_dim = 4  
class_num = 2  
N = 14  
X_P = ['Sunny', 'Hot', 'Normal','False']
print("classify the item: ")
print(X_P)
lle,prior=freq_num (N,data,features_dim,X_P)
print("\nLikelihood of Evidence’ : ")
print(lle)
print("\nPrior frequencies: ")
print(prior)

post_prob=evid_terms(class_num,features_dim,lle,prior,N)
np.set_printoptions(3)
print("\nEvidence terms: ")
print(post_prob)
evidence = np.sum(post_prob)
probs = np.zeros(class_num, dtype=np.float32)
for k in range(class_num):
    probs[k] = post_prob[k] / evidence

print("\nFinal classes probabilities: ")
print(probs)
pc = np.argmax(probs)
print("\nPredicted class: ")
print(pc)

