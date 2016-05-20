
# coding: utf-8

# In[ ]:

#------------------------- Global Initialization Code-------------------#
import numpy as np
import mltools as ml
import matplotlib.pyplot as plt # use matplotlib for plotting with inline plots
import random
import mltools.linear
import time #use time for getting the time for an execution
from sklearn.svm import SVR
from sklearn import tree
from sklearn.metrics import mean_squared_error as mse
get_ipython().magic(u'matplotlib inline')

# get training data
X = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
Xe1 = np.genfromtxt("data/kaggle.X1.test.txt", delimiter = ', ')

# split the training data?
Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)


# In[ ]:

# Create a class with basic interface
class Ensemble:
    def __init__(self):
        self.learners = []
        self.phis = []
    
    def add(self, learner, phi=None):
        self.learners.append(learner)
        if phi == None:
            self.phis.append(lambda x : x)
        else:
            self.phis.append(phi)
    
    def predict(self, X):
        predictions = []
        for learner,phi in zip(self.learners, self.phis):
            predictions.append(learner.predict(phi(X))) # Predict using each learner once
        prediction = np.mean(predictions, axis=0) # Average the the predictions
        return prediction.ravel()

# Instantiate the class
ensemble = Ensemble()


# In[ ]:

#--------------------------------Build the learners ------------------------#


# In[ ]:

### Xinglong Bai's learners
knn = ml.knn.knnRegress(Xtr[:, 0:4], Ytr) #It's currently too slow, don't use it for ensemble now
knn.K = 5

stop = 0
#for i in range(Xte.shape[0] // 200):
#    Vmse = knn.mse(Xte[i*200:(i+1)*200,0:4], Yte[i*200:(i+1)*200])
#    print("data sets ", i)
#    print("Vmse: ", Vmse)

m, n = Xtr.shape
nUse = m


###K nearest neighbor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

## select k
K = [1, 5, 25, 40, 75, 80, 85, 90, 100, 120]
knnpredict = []
knnVMse = []
for k in K:
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(Xtr[:, 0:4], Ytr)
    knnpredict.append(knn.predict(Xte[:, 0:4]))
    
for p in knnpredict:
    knnVMse.append(mean_squared_error(Yte, p))
for i in range(len(knnVMse)):
    print("k =", K[i], " VMSE =", knnVMse[i])

bestK = K[knnVMse.index(min(knnVMse))]
print("Best k =", bestK)
"""Best k = 80"""
##feature selection
knnpredict = []
knnVMse = []
xs = [i for i in range(n)]
for i in range(n):
    knn = KNeighborsRegressor(n_neighbors = bestK)
    knn.fit(Xtr[:, i].reshape(Xtr.shape[0], 1), Ytr)
    p = knn.predict(Xte[:, i].reshape(Xte.shape[0], 1))
    knnpredict.append(p)
    knnVMse.append(mean_squared_error(Yte, p))
    
for i in range(n):
    print("x" + str(i), " VMSE =", knnVMse[i])

zipped = zip(xs, knnVMse)
featureRank = sorted(zipped, key = lambda x: x[1])
print(featureRank[0:10])

## train with best K and best 4 features
features = []
for i in range(30):
    features.append(featureRank[i][0])

selectedf = tuple(features)

phi = lambda X: X[:, (2, 23, 3, 5, 27, 9, 8, 19, 10, 15, 7, 55, 83, 6, 84, 4, 0, 67, 63, 11, 87, 88, 35, 71, 39, 43, 40, 59, 42, 36)]


knn = KNeighborsRegressor(n_neighbors = bestK)
knn.fit(Xtr[:, selectedf], Ytr)
p = knn.predict(Xte[:, selectedf])
ptr = knn.predict(Xtr[:, selectedf])
VMSE = mean_squared_error(Yte, p)
Tmse = mean_squared_error(Ytr, ptr)
"""             selectedf                   selectedf  """
"""50 features: 0.45138, 0.4515  40 features: 0.4556 0.453370153061
   35 features: 0.45016, 0.455   30 features: 0.45016, 0.455
   20 features: 0.45492, 0.4570  60 features: 0.47711, 0.477250321814
   15 features: 0.46504
   2  features: 0.52485
"""

print("k = " + str(bestK), "features = ", selectedf, "Vmse =", VMSE)
print("k = " + str(bestK), "features = ", selectedf, "Tmse =", Tmse)

"""
KNN learner use guide:
phi = lambda X: X[:, (2, 23, 3, 5, 27, 9, 8, 19, 10, 15, 7, 55, 83, 6, 84, 4, 0, 67, 63, 11, 87, 88, 35, 71, 39, 43, 40, 59, 42, 36)]

predict = knn.predict(phi(X))
"""
ensemble.add(knn, phi)



# In[ ]:

m, n = Xtr.shape
nUse = m

ind = np.ceil(m*np.random.rand(nUse)).astype(int)
ind = ind-1
Xa, Ya = Xtr[ind, :], Ytr[ind]
lr = ml.linear.linearRegress(Xa, Ya) # Use this linear regression learner at first
ensemble.add(lr)

Tmse = lr.mse(Xtr, Ytr)
Vmse = lr.mse(Xte, Yte)

print("Tmse: ", Tmse)
print("Vmse: ", Vmse)


# In[ ]:

### Minjun Yu's SVM regression learners
#
# Experimental things ommited for now
#
## produce the actual learner
svr_rbf = SVR(kernel="rbf", C=10, gamma=0.1)
svr_learner = svr_rbf.fit(Xtr, Ytr)
ensemble.add(svr_learner)


# In[ ]:

### Andrew Fischer's learners
tree_learner = Ensemble()

# Find results from bagged tree
for i in range(0, 50):
    time_start = time.time()
    x,y = ml.bootstrapData(X, Y, len(X))
    tl = tree.DecisionTreeRegressor(max_features=50)
    tl.fit(x, y)
    tree_learner.add(tl)
    time_end = time.time()
    print("Iteration=" + str(i) + ", seconds=" + str(time_end - time_start))

ensemble.add(tree_learner)


# In[ ]:

#--------------------------------- Ensemble ---------------------------------#
# store the learners we have in to a list, index them and then make predictions
Yhat = ensemble.predict(Xe1)


# In[ ]:

#--------------------------------- Output and evaluate data ----------------#
# Save Yhat in required format
f=open("kaggle.Y.test-" + str(time.time()) + ".txt", "w")
f.write("ID,Prediction\n")
for i,yi in enumerate(Yhat):
    f.write("{},{}\n".format(i+1,yi))
f.close()

