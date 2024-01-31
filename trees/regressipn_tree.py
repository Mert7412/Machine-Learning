import numpy as np
import pandas as pd

data = pd.read_csv(r"trees\diabetes.csv").values

class node:
    def __init__(self,subnodes=None,threshold=None,feautreindex=None,value=None):
        self.subnodes = subnodes
        self.threshold = threshold
        self.featureindex = feautreindex
        self.value = value

class regressiontree:
    def __init__(self):
        self.maintree = None
    
    def rss(self,y1,y2):
        rss = np.sum((y1-y2)**2)
        return rss
    
    def split(self,x,y,threshold,featureindex):
        spx = []
        spy = []
        for i in range(len(threshold)-1):
            subspx = []
            subspy = []
            for j in range(len(x)):
                if threshold[i] <= x[j][featureindex] < threshold[i+1] or x[j][featureindex] == threshold[-1]:
                    subspx.append(x[j])
                    subspy.append(y[j])
            spx.append(np.array(subspx,dtype=object))
            spy.append(np.array(subspy,dtype=object))
        return np.array(spx,dtype=object),np.array(spy,dtype=object)

    def calculatemean(self,x,featureindex):
        means = []
        x = np.unique(x.T[featureindex])
        for i in range(len(x)-1):
            mean = (x[i]+x[i+1])/2
            means.append(mean)
        return means
    
    def bestsplit(self,x,y):
        taf = []
        mse = []
        for i in range(len(x[0])):
            fmin = min(x.T[i])
            fmax = max(x.T[i])
            means = self.calculatemean(x,i)
            for m in means:
                splx,sply = self.split(x,y,[fmin,m,fmax],i)
                meofsp = [np.mean(i) for i in sply]
                mse1 = self.rss(sply[0],meofsp[0])
                mse2 = self.rss(sply[1],meofsp[1])
                smse = mse1+mse2
                mse.append(smse)
                taf.append([[fmin,m,fmax],i])
        minmse = np.argmin(mse)
        besttaf = taf[minmse]
        bestthreshold = besttaf[0]
        bestfeain = besttaf[1]
        subdata = self.split(x,y,bestthreshold,bestfeain)
        return subdata,bestthreshold,bestfeain
        
    
    def tree(self,x,y,maxdepth,min_sample_split,depht=0):
        if depht <= maxdepth:
            value = np.mean(y)
            datasplit =self.bestsplit(x,y)
            datax = datasplit[0][0]
            datay = datasplit[0][1]
            
            subtree = []
            
            for i in range(len(datax)):
                if len(datax[i]) > min_sample_split:
                    subnode = self.tree(datax[i],datay[i],maxdepth,min_sample_split,depht+1)
                    subtree.append(subnode)
            return node(subtree,datasplit[1],datasplit[2],value)
    
    def fit_tree(self,x,y,maxdepth = 100,min_sample_split =20):
        self.maintree = self.tree(x,y,maxdepth,min_sample_split)
    
    def predict(self,feature,tree):
        threshold =tree.threshold
        featureindex = tree.featureindex
        nodevalue = tree.value
        for i in range(len(tree.subnodes)):
            if tree.subnodes[i] == None:
                break
            elif threshold[i] <= feature[featureindex] < threshold[i+1] or feature[featureindex] == threshold[i+1]:
                nodevalue = self.predict(feature,tree.subnodes[i])
        return nodevalue
    
    def make_prediction(self,test_feature):
        prediction = []
        for f in test_feature:
            prediction.append(self.predict(f,self.maintree))
        return prediction

    def r2score(self,ytest,ypred):
        r2 = 1 - (self.rss(ytest,ypred)/(np.sum((ytest-np.mean(ytest))**2)))
        return r2
x = data.T[:-1].T
y = data.T[-1]

a = regressiontree()
a.fit_tree(x,y)
b = a.make_prediction(x)
print(a.r2score(y,b))

