import numpy as np
import pandas as pd

"""data = pd.read_csv(r"trees\wineqr.csv").values
dwata = data.drop("quality",axis=1).values"""
"""quality = data["quality"].values"""
"""datat = pd.read_csv(r"trees\wineqr.csv").values.T"""
"""da1  = pd.read_csv(r"trees\diabetes.csv").values
f1 = da1.T[:-1].T
"""
class node:
    def __init__(self,subdata=None,threshold=None,value=None,featurei=None,leaf = False):
        self.subdata = subdata
        self.featurei = featurei
        self.threshold = threshold
        self.value = value
        self.leaf = leaf

class desiciontree:
    def __init__(self):  
        self.maintree = None

    def probability(self,label):
        lenghtoflabel = len(label)
        uniqe = np.unique(label,return_counts=True)
        prob = uniqe[1]/lenghtoflabel
        return prob

    def calculate_entropy(self,probabilities):
        entropy = -np.sum(([probabilities]*np.log([probabilities])),axis=1)
        return entropy

    def multiple_entropy(self,subfeatureprob,entropy):
        mentropy = np.sum(subfeatureprob*entropy)
        return mentropy

    def informationgain(self,entropy_1,entropy_2):
        infoga = entropy_1 - entropy_2
        return infoga

    def splitting(self,data,threshold,featureindex,min_sample_split):
        datat = data.T
        feature = datat[:len(datat)-1].T
        label = datat[-1]
        lenghtofthreshold = len(threshold)
        split = []
        for i in range(lenghtofthreshold-1):
            subsplit = []
            a = 0
            for f in feature:
                if threshold[i] < f[featureindex] < threshold[i+1]:
                    subsplit.append(np.concatenate((f,[label[a]])))
                a +=1
            if len(subsplit) <= min_sample_split:
                continue
            split.append(np.array(subsplit))
        return np.array(split,dtype=object)


    def calculatemeanofvalues(self,data):
        means = []
        datat = data.T
        features = datat[:len(datat)-1].T
        qual = datat[-1]
        featurest = datat[:len(datat)-1]
        for i in range(len(features[0])):
            meanseachfeatures = []
            sortfeatures = np.sort(featurest[i])
            for f in range(len(features)-1):
                mean = (sortfeatures[f]+sortfeatures[f+1])/2
                meanseachfeatures.append(mean)
            means.append(meanseachfeatures)
        return means

    def bestsplit(self,data,minsamplesplit):
        datat = data.T
        features = datat[:len(datat)-1].T
        label = datat[-1]
        featurest = datat[:len(datat)-1]
        minfea = np.amin(featurest,axis=1)
        maxfea = np.amax(featurest,axis=1)
        probabilitiesoflabel = self.probability(label)
        entropy_1 = self.calculate_entropy(probabilitiesoflabel)
        informationgain = []
        means = self.calculatemeanofvalues(data)
        for i in range(len(features[0])):
            infgainforeachm = []

            for m in means[i]:
                splitfea = self.splitting(data,[minfea[i],m,maxfea[i]],i,minsamplesplit)
                entropies = np.array([])
                lenoffea = 0
                lenofsplit = np.array([])
                for s in splitfea:
                    prob = self.probability(s.T[-1])
                    entropy_temporary = self.calculate_entropy(prob)
                    entropies = np.append(entropies,entropy_temporary)
                    lenoffea += len(s)
                    lenofsplit = np.append(lenofsplit,len(s))
                subfeaprob = lenofsplit/lenoffea
                entropy_2 = self.multiple_entropy(subfeaprob,entropies)
                infogain = self.informationgain(entropy_1,entropy_2)
                infgainforeachm.append(infogain)
            informationgain.append(infgainforeachm)
        maxgainseveryfea = np.amax(informationgain,axis=1)
        maxgainindex = np.argmax(maxgainseveryfea)
        maxgain = np.array(informationgain[maxgainindex])
        best_threshold_index = np.where(maxgain == maxgainseveryfea[maxgainindex][0])[0][0]
    
        best_threshold = [minfea[maxgainindex],means[maxgainindex][best_threshold_index],maxfea[maxgainindex]]
  
        subdata = self.splitting(data,best_threshold,maxgainindex,minsamplesplit)
        return subdata, maxgainindex, best_threshold
    
    def calculateleafvalue(self,data):
        label = data.T[-1]
        uniqe = np.unique(label,return_counts=True)
        maxi = np.argmax(uniqe[1])
        return uniqe[0][maxi]
    
    def tree(self,data,depth,maxdepth,minsamplesplit):
        if depth < maxdepth:
            leafval = self.calculateleafvalue(data)
            splidat = self.bestsplit(data,minsamplesplit)       
            subnode = []
            for s in splidat[0]:
                subtree = self.tree(s,depth+1,maxdepth,minsamplesplit)
                subnode.append(subtree)
            return node(subnode,splidat[2],leafval,splidat[1])
    
    def fit_tree(self,data,depth=0,maxdepth=100,minsamplesplit = 4):
        self.maintree = self.tree(data,depth,maxdepth,minsamplesplit)

    
    def predict(self,feature,tree):
        threshold =tree.threshold
        featureindex = tree.featurei
        nodevalue = tree.value
        for i in range(len(tree.subdata)):
            if tree.subdata[i] == None:
                break
            elif threshold[i] <= feature[featureindex] < threshold[i+1] or feature[featureindex] == threshold[i+1]:
                nodevalue = self.predict(feature,tree.subdata[i])
        return nodevalue
    
    def make_prediction(self,test_feature):
        prediction = []
        for f in test_feature:
            prediction.append(self.predict(f,self.maintree))
        return prediction

    def confusionmatrix(self,y_test,y_pred):
        numberofclass = len(np.unique(y_test)) 
        clasval = np.unique(y_test)
        tp,tn,fp,fn = np.zeros(numberofclass),np.zeros(numberofclass),np.zeros(numberofclass),np.zeros(numberofclass)
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                for c in range(numberofclass):
                    if clasval[c] == y_pred[i]:
                        tp[c] += 1
                    else:
                        tn[c] +=1
            else:
                for d in range(numberofclass):
                    if clasval[d] == y_pred[i]:
                        fp[d] += 1
                    else:
                        fn[d] +=1
        recall = tp/(tp+fn)
        presicion = tp /( tp+fp)
        f1 = (2 * recall*presicion)/(recall+presicion)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        return f"recall:{recall}, presicion:{presicion}, f1-score:{f1}, accuracy:{accuracy}"
      
"""train = da1[:500]
test = da1[500:]
trea = test.T[:-1].T
trees = test.T[-1]

tree = desiciontree()

tree.fit_tree(train)

p = tree.make_prediction(trea)

print(tree.confusionmatrix(trees,p))"""
"""
dt = data[:1300]
tt = data[1300:]
sdfs =tt.T[:-1].T
trefsd = tt.T[-1]

tree = desiciontree()
tree.fit_tree(dt,maxdepth=10)
b = tree.make_prediction(sdfs)
print(tree.confusionmatrix(trefsd,b))
"""
