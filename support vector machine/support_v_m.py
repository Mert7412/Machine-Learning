import pandas as pd
import numpy as np

data = pd.read_csv(r"support vector machine\diabetes.csv").values
x = data.T[:-1].T
y = data.T[-1]
y[y == 0] = -1
x = (x - np.mean(x))/ np.std(x)

class svm:
    def __init__(self,c):
        self.w = np.zeros(len(x[0]))
        self.b = np.zeros(len(x))
        self.alpha = np.zeros(len(x))
        self.tol = 0.001
        self.C = c
        self.KM = self.K(x,x)

    def function1(self,x,w,b):
        f = np.sum(w*x) + b
        return f
      
    def function2(self,alpha,y,b):
        f = np.sum(alpha*y*self.KM,axis=1)+b
        return f
    
    def hingeloss(self,x,y):
        f = self.function1(x,self.w,self.b)
        loss = np.maximum(0,1-y*f)
        return np.sum(loss)
    
    def derhingeloss(self,x,y,tetha):
        f = self.function1(x,self.w,self.b)
        lossw = 0
        lossb = 0
        for i in range(len(x)):
            if y[i]*f[i] <1:
                 lossw += (2*self.w*tetha) - (y[i]*x[i])
                 lossb += (2*self.b*tetha)  - y[i]
        return lossw , lossb

    def smo(self,iteration):
        nonchangediteration = 0
        for qqqqq in range(iteration):
            f = self.function2(self.alpha,y,self.b)
            oldalpha = np.copy(self.alpha)
            for i in range(len(x)):
                ai = self.alpha[i]
                
                fi = f[i]
                yi = y[i]
                Ei = fi - yi
            
                if (yi*Ei < -self.tol and ai < self.C) or (yi*Ei > self.tol and ai > 0) :
                    
                    if len(np.nonzero(a.alpha)[0]) + len(a.alpha[a.alpha != self.C]) >= 1:   
                        j = self.second_multiplier(f)
                        if i == j:
                            continue
                        oldai = np.copy(ai)
                        oldaj = np.copy(self.alpha[j])
                        fj = f[j]
                        yj = y[j]
                        Ej = fj - yj
                        if y[i] != y[j]:
                            L = max(0,self.alpha[j]-self.alpha[i])
                            H = min(self.C,self.C+self.alpha[j]-self.alpha[i])
                        else:
                            L = max(0,self.alpha[j]+self.alpha[i]-self.C)
                            H = min(self.C,self.alpha[j]+self.alpha[i])
                        if L == H:
                            continue
                        epsilon = 1e-6
                        eta = -(self.KM[i,i]+self.KM[j,j]+-2*self.KM[i,j])
                        if eta >= 0:
                            continue
                        
                        self.alpha[j] -= (y[j]*(Ei-Ej))/eta
                        self.alpha[j] = np.clip(self.alpha[j],L,H)
                        if abs(self.alpha[j]- oldaj) < 1e-5:
                            continue
                        
                        
                        self.alpha[i] += y[i]*y[j]*(oldaj-self.alpha[j])
                        self.b = self.compute_b(Ei,Ej,i,j,oldai,oldaj)
                        self.w = self.w + y[i]*(self.alpha[i]-oldai)*x[i]+y[j]*(self.alpha[j]-oldaj)*x[j]
            changedalp = abs(self.alpha - oldalpha)
            if np.count_nonzero(changedalp > 1e-5) == 0:
                nonchangediteration += 1
            else:
                nonchangediteration = 0
            if nonchangediteration == 6:
                break
            
            print(self.accuracy(self.predict(x),y))

    def K(self,x1,x2,kernel = "linear",c= 1,d=2,gamma = 0.1):
        if kernel == "linear":
            k = np.dot(x1,x2.T)
            return k 
        elif kernel == "polynomial":
            k = (np.dot(x1,x2.T)+c)**d
            return k
      
    def compute_b(self,Ei,Ej,i,j,oldai,oldaj):
        b1 = -(Ei+y[i]*(self.alpha[i]-oldai)*(self.KM[i,i])+y[j]*(self.alpha[j]-oldaj)*(self.KM[i,j])-self.b)
        b2 = -(Ej+y[i]*(self.alpha[i]-oldai)*(self.KM[i,j])+y[j]*(self.alpha[j]-oldaj)*(self.KM[j,j])-self.b)
        if 0 < self.alpha[i] < self.C:
            return b1
        elif 0 < self.alpha[j] < self.C:
            return b2
        else:
            b = (b1 + b2) / 2.0 
            return b

    def second_multiplier(self,f):
        l = []
        for i in range(len(x)):
            error = f[i] - y[i]
            l.append(abs(error))     
        return np.argmax(l)

    def fit1(self,x,y,iteration,alpha,tetha):
        for i in range(iteration):
            ls = self.derhingeloss(x,y,tetha)
            self.w -= alpha*ls[0]
            self.b -= alpha*ls[1]
            sdasd = self.predict(x)
            print(self.accuracy(sdasd,y))
              
    def fit(self,x,y,iteration,alpha,tetha): 
        for j in range(iteration):
            f = self.function1(x,self.w,self.b)
            for i in range(len(x)):
                if y[i]*f[i] >=1:
                    self.w -= alpha * (2*self.w*tetha)
                    self.b -= alpha * (2*self.b*tetha)
                else:
                    self.w -= alpha*(2*self.w*tetha - y[i]*x[i])
                    self.b -= alpha*(2*self.b*tetha  - y[i])

    def predict(self,x):
        p = np.sign(self.function2(self.alpha,y,self.b))
        return p
    
    def accuracy(self,predicted_labels, actual_labels):
            diff = predicted_labels - actual_labels
            return  1 - (float(np.count_nonzero(diff)) / len(diff))
    
    def scale(self,x):
            meanx = np.mean(x,axis=0)
            stdx = np.std(x,axis=0)
            return (x-meanx)/stdx
    
a = svm(0.1)
x = a.scale(x)
a.smo(1000)
yp = a.predict(x)








