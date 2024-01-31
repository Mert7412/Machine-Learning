import pandas as pd
import numpy as np

data = pd.read_excel(r"regression\data\Folds5x2_pp.xlsx").values
y = data.T[-1]
y1 = y[:round(len(y)*0.8)]
y2 = y[round(len(y)*0.8):]
x = data[:,:-1]
x1 = x[:round(len(x)*0.8)]
x2 = x[round(len(x)*0.8):]

class linearregression:
    def __init__(self,x,y):
        self.x = self.scale(x)
        self.y = y
        self.coefficent = np.zeros((len(self.x[0])+1))  
        
    def scale(self,x):
        meanx = np.mean(x,axis=0)
        stdx = np.std(x,axis=0)
        return (x-meanx)/stdx
        
    def function(self,x):       
        y_pred = np.array((((self.coefficent[1:]*x)+(self.coefficent[0]/len(x[0])))))        
        return np.sum(y_pred,axis=1)

    def mse(self):
        mse = np.mean(((self.y - self.function(self.x))**2))
        return mse 
     
    def dmse(self):   
        dmse1 = np.sum((-2/len(self.x))*(self.y - self.function(self.x)))
        dmse2 = np.sum(((-2/len(self.x))*(self.y - self.function(self.x))).reshape((len(self.x),1))*self.x,axis=0)
        return np.hstack((dmse1,dmse2))

    def gradientdesc(self,learningrate,iter):
        premse = 0    
        for i in range(iter):
            self.coefficent = self.coefficent - (self.dmse()* learningrate)        
            if np.abs(self.mse() -premse) < 10e-06:
                break     
            print(f"iter: {i+1}, mse: {self.mse()}, r2_score:{self.r2_score(self.x,self.y)}")
            premse =  self.mse()
    
    def predict(self,x_test,y_test):
        x_test = self.scale(x_test)
        return self.function(x_test) ,self.r2_score(x_test,y_test)
 
    def r2_score(self,x_test,y_test):
        x_test = self.scale(x_test)
        score = 1 - ((np.sum((y_test - self.function(x_test))**2))/(np.sum((y_test-np.mean(y_test))**2)))
        return score  
   

        
a = linearregression(x1,y1)
  
a.gradientdesc(0.001,10000)

print(a.predict(x2,y2))
    
    
    


