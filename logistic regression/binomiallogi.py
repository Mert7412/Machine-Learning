import pandas as pd
import numpy as np

data = pd.read_csv(r"logistic regression\data2.csv")
x = data.drop(["label","aa"],axis=1).values
y = data["label"].values

class binomiallogistic:
    def __init__(self,x,y):
        self.x = self.scale(x)
        self.y = y
        self.coefficent = np.zeros((len(self.x[0])+1))  
        
    def scale(self,x):
        meanx = np.mean(x,axis=0)
        stdx = np.std(x,axis=0)
        return (x-meanx)/stdx
    
    def function(self,x):
        y =np.array(((self.coefficent[1:]*x)+(self.coefficent[0]/len(x[0]))))
        return np.sum(y,axis=1)
    
    def sigmoidfunc(self,x):
        p = 1/(1+np.exp(-(self.function(x))))
        return p
    
    def costfunction(self,x,y):
        cost = np.mean(-y*np.log(self.sigmoidfunc(x)) - (1-y)*np.log(1-self.sigmoidfunc(x)))
        return cost

    def derivatecostfunction(self,x,y):
        cost1 = np.mean(-((y*(np.exp(-self.function(x)))) + (1 - y)*((1+np.exp(-self.function(x))-np.exp(-self.function(x)))/np.exp(-self.function(x)))))
        cost2 = np.mean(-((y*(np.exp(-self.function(x)))*x.T) + (1 - y)*((1+np.exp(-self.function(x))-np.exp(-self.function(x))*x.T)/np.exp(-self.function(x)))),axis=1)
        return np.hstack((cost1,cost2))
    
    def decisionboundary(self,proba):
        list= []
        for prob in proba:
            if prob >= 0.3:
                list.append(1)
            else:
                list.append(0)
        return np.array(list)

    def accuracy(self,predicted_labels, actual_labels):
        diff = predicted_labels - actual_labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

    def gradientdescent(self,iter,learningrate):
        premse = 0
        for i in range(iter):
            self.coefficent = self.coefficent - (self.derivatecostfunction(self.x,self.y)*learningrate)
            if np.abs(self.costfunction(self.x,self.y)-premse)< 10e-06:
                break
            premse = self.costfunction(self.x,self.y)
            
            print(f"iter:{i+1}  cost:{self.costfunction(self.x,self.y)}  accuary:{self.accuracy(self.decisionboundary(self.sigmoidfunc(self.x)),self.y)}")


a = binomiallogistic(x,y)
a.gradientdescent(10000,0.001)
