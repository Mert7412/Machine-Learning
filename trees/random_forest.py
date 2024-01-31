import numpy as np
import desicion_tree_algorithm as dt
import pandas as pd
data = pd.read_csv(r"trees\diabetes.csv")

class randomforest:
    def __init__(self,noftrees,data,datasize = None):
        self.noftrees = noftrees
        self.data = data
        self.trees = []
        self.datasize = datasize
        if self.datasize == None:
            self.datasize = len(self.data)
        for i in range(noftrees):
            randomdata = pd.DataFrame.sample(data,n=self.datasize,replace=True)
            tree = dt.desiciontree()
            tree.fit_tree(randomdata.values)
            self.trees.append(tree.maintree)

    def predict(self,feature):
            values = []
            for tree in self.trees:
                 value = dt.desiciontree().predict(feature,tree)
                 values.append(value)
            uniqeval = np.unique(values,return_counts=True)
            maxindex = np.argmax(uniqeval[1])
            return uniqeval[0][maxindex]

    def make_prediction(self,test_data):
         prediction = []
         for x in test_data:
              prediction.append(self.predict(x))
         return prediction
              

da1 = data.values
da1_x = da1.T[:-1].T
da1_y = da1.T[-1]

b = randomforest(3,data)
p = b.make_prediction(da1_x)
print(dt.desiciontree().confusionmatrix(da1_y,p))







