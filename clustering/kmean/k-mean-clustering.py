import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
np.seterr(divide="ignore",invalid="ignore")
np.set_printoptions(suppress=True)
data = loadmat(r"clustering\kmean\data\diamond9.mat")["data"]


class kmean():
    def __init__(self,data,numberofcluster):
        self.data = data
        self.dlen = len(data)
        self.k = numberofcluster
        self.x = len(data[0])
        self.centroids = self.centroidsinitia()
        
    def centroidsinitia(self):
        means = np.zeros([self.k,self.x])

        for i in range(len(means)):
            rand = np.random.randint(0,self.dlen)
            means[i] = self.data[rand]
        return means
    
    def euclidiandistance(self,centroids):
        distance = []
        for c in centroids:
            sumofd = np.sqrt(np.sum(((self.data - c)**2),axis=1))
            distance.append(sumofd)
        return np.array(distance)
    
    def classify(self,centroids):
        index = np.argmin(self.euclidiandistance(centroids),axis=0)
        return index

    def clustersize(self,classify):
        sizelist = np.zeros(self.k)
        for c in classify:
            sizelist[c] += 1
        return sizelist

    def updatecentroids(self,index,size):
        scentroids = np.zeros((self.k,self.x))
        for i in range(self.dlen):
            scentroids[index[i]] += self.data[i]
        self.centroids = (scentroids.T/size).T
    
    def fit(self,iter):
        for i in range(iter):
            index = self.classify(self.centroids)
            size = self.clustersize(index)
            self.updatecentroids(index,size)

a = kmean(data,9)  

a.fit(100)
 
tcentr = a.centroids.T   

plt.scatter(data.T[0],data.T[1])
plt.scatter(tcentr[0],tcentr[1],color = "black")

plt.show()
 











